#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
extern int target_length;
extern char *target_folder;
extern int softmax_Temperature;


void train_KD_regressor_pretrained(char *solver, char *cfgfile, char *cfgfile_valid, char *weightfile, char *cfgfile_pretrained, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(solver);
    char *backup_directory = option_find_str(options, "backup", "models");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int test_interval = option_find_int(options, "test_interval", 500);
    int snapshot = option_find_int(options, "snapshot", 1000);
    target_folder = option_find_str(options, "target_folder", "targets");
    target_length = option_find_int(options, "target_length", 1);
    softmax_Temperature = option_find_int(options, "softmax_Temperature", 1);
    fprintf(stderr,"\ntarget_folder = %s\n", target_folder);

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_pretrained(cfgfile, weightfile, cfgfile_pretrained, clear);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];
    if(net->truths != target_length) error("net->truths != target_length");

    int imgs = net->batch * net->subdivisions * ngpus;
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    fprintf(stderr,"\nNo of training samples = %d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = REGRESSION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    fprintf(stderr,"\nTraining..");
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        if(get_current_batch(net)==1){
            fprintf(stderr,"\niteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
        }

        // Validation and Display
        if(get_current_batch(net)%test_interval == 0){  // Test_interval
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            fprintf(stderr,"\niteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);                        
        }
        
        // Snapshot 
        if(get_current_batch(net)%snapshot == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%ld.weights",backup_directory,base, get_current_batch(net));
            save_weights(net, buff);
        }
    }
    pthread_join(load_thread, 0);
    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}


void train_KD_regressor(char *solver, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(solver);
    char *backup_directory = option_find_str(options, "backup", "models");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int test_interval = option_find_int(options, "test_interval", 500);
    int snapshot = option_find_int(options, "snapshot", 1000);
    target_folder = option_find_str(options, "target_folder", "targets");
    target_length = option_find_int(options, "target_length", 1);
    softmax_Temperature = option_find_int(options, "softmax_Temperature", 1);
    fprintf(stderr,"\ntarget_folder = %s\n", target_folder);

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];
    if(net->truths != target_length) error("net->truths != target_length");

    int imgs = net->batch * net->subdivisions * ngpus;
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    fprintf(stderr,"\nNo of training samples = %d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = REGRESSION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    fprintf(stderr,"\nTraining..");
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        if(get_current_batch(net)==1){
            fprintf(stderr,"\niteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
        }

        // Validation and Display
        if(get_current_batch(net)%test_interval == 0){  // Test_interval
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            fprintf(stderr,"\niteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);                        
        }
        
        // Snapshot 
        if(get_current_batch(net)%snapshot == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%ld.weights",backup_directory,base, get_current_batch(net));
            save_weights(net, buff);
        }
    }
    pthread_join(load_thread, 0);
    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}


void run_KD_regressor(int argc, char **argv)
{
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");

    char *solver; char *cfg; char *cfg_valid; char *weights; char *cfg_pretrained; char *filename; char *layer_s; int layer;
    
    fprintf(stderr,"\n\nFormat1: darknet kd_regressor train solver train_cfg");
    fprintf(stderr,"\n\nFormat2: darknet kd_regressor train solver train_cfg pretrained_weights pretrained_cfg");
    
    if(0==strcmp(argv[2], "train"))   // Training arguments
    {
    // Format1: darknet kd_regressor train solver train_cfg //
    // Format2: darknet kd_regressor train solver train_cfg pretrained_weights pretrained_cfg //    

        // Format 1 (Training from scratch)
        if(argc < 5) error("\nInsufficient no of arguments(<6)");
        solver = argv[3];   // Solver
        cfg = argv[4];      // train_cfg
        
        // Format 2 (Training from pretrained weights)
        if(argc > 5 && argc <8) error("\nInsufficient no of arguments(<8)");
        cfg_valid = (argc > 5) ? argv[5]: 0;      // test_cfg    
        weights = (argc > 6) ? argv[6]: 0;    // pretrained_weights
        cfg_pretrained = (argc > 7) ? argv[7]: 0; // pretrained_cfg        
        filename = (argc > 8) ? argv[8]: 0;
        layer_s = (argc > 9) ? argv[9]: 0;
        
    }
    else     // Valid/Predict/Test arguments
    {
        error("\nIncorrect arguments");
    }
    
    
    if(0==strcmp(argv[2], "train")) {
        if(argc ==5)
            train_KD_regressor(solver, cfg, weights, gpus, ngpus, clear);
        else
            train_KD_regressor_pretrained(solver, cfg, cfg_valid, weights, cfg_pretrained, gpus, ngpus, clear);
    }
}


