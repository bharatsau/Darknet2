#include "darknet.h"
extern int partial_weights;
extern int target_length;
extern char *target_folder;
extern int multiloss_cost;

void train_multiloss_detector_pretrained(char *solver, char *cfgfile, char *cfgfile_valid, char *weightfile, char *cfgfile_pretrained, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(solver);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "models");
    target_folder = option_find_str(options, "target_folder", "targets");
    target_length = option_find_int(options, "target_length", 1);
    int test_interval = option_find_int(options, "test_interval", 500);
    int snapshot = option_find_int(options, "snapshot", 1000);
    int partial_weights = option_find_int(options, "partial_weights", 0);

    srand(time(0));
    char *base = basecfg(cfgfile);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif        
        if(partial_weights){
            nets[i] = load_network_partial_weights(cfgfile, weightfile, cfgfile_pretrained, clear);        
        }
        else{
            nets[i] = load_network_pretrained(cfgfile, weightfile, cfgfile_pretrained, clear);
        }
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data train, buffer;

    layer l = net->layers[net->n - 1];
    if(l.type != YOLO) error("\nFinal Layer should be a Detection layer(Yolo/Region)..\n\n");

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);
    fprintf(stderr,"\n\nNo of training samples = %d\n", plist->size);    

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_REGRESSION_DATA;
    args.threads = 32;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    fprintf(stderr, "\nCurrent_batch= %d\n", get_current_batch(net));

    while(get_current_batch(net) < net->max_batches){

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_multiloss_network(net, train);
        } else {
            error("\nDoes not support multi-gpu training for this.\n");
        }
#else
        loss = train_multiloss_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        // Snapshot
        if(get_current_batch(net)%snapshot == 0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            fprintf(stderr,"iteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }

        // Validation and Display
        if(get_current_batch(net)%test_interval == 0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            fprintf(stderr,"iteration %ld: train_loss = %f, avg_train_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            //validate_detector_recall(solver, cfgfile_valid, buff, 0.24, 0.5);
        }
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void train_multiloss_detector(char *solver, char *cfgfile, char *cfgfile_valid, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(solver);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "models");
    target_folder = option_find_str(options, "target_folder", "targets");
    target_length = option_find_int(options, "target_length", 1);
    int test_interval = option_find_int(options, "test_interval", 500);
    int snapshot = option_find_int(options, "snapshot", 1000);

    srand(time(0));
    char *base = basecfg(cfgfile);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

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
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data train, buffer;

    layer l = net->layers[net->n - 1];
    if(l.type != YOLO) error("\nFinal Layer should be a Detection layer(Yolo/Region)..\n\n");

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);
    fprintf(stderr,"No of training samples = %d\n", plist->size);    

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_REGRESSION_DATA;
    args.threads = 32;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    fprintf(stderr, "\nCurrent_batch= %d\n", get_current_batch(net));

    while(get_current_batch(net) < net->max_batches){

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_multiloss_network(net, train);
        } else {
            error("\nDoes not support multi-gpu training for this.\n");
        }
#else
        loss = train_multiloss_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        // Snapshot
        if(get_current_batch(net)%snapshot == 0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            fprintf(stderr,"iteration %ld: loss = %f, avg_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }

        // Validation and Display
        if(get_current_batch(net)%test_interval == 0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            fprintf(stderr,"iteration %ld: train_loss = %f, avg_train_loss(avg_loss*.9 + loss*.1) = %f\n", get_current_batch(net), loss, avg_loss);
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            //validate_detector_recall(solver, cfgfile_valid, buff, 0.24, 0.5);
        }
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}



void run_multiloss_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);
    
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);

    char *solver; char *cfg; char *cfg_valid; char *weights; char *cfg_pretrained; char *filename; char *layer_s;

    if(0==strcmp(argv[2], "train"))   // Training arguments
    {
    // Format1: darknet classifier/detector train solver train_cfg test_cfg //
    // Format2: darknet classifier/detector train solver train_cfg test_cfg pretrained_weights pretrained_cfg  //    
        // Format 1 (Training from scratch)
        if(argc < 6) error("\nInsufficient no of arguments(<5)");
        solver = argv[3];   // Solver
        cfg = argv[4];      // train_cfg
        cfg_valid = argv[5];  // test_cfg            
        
        // Format 2 (Training from pretrained weights)
        if(argc > 6 && argc <8) error("\nInsufficient no of arguments(<8)");        
        weights = (argc > 6) ? argv[6]: 0;    // pretrained_weights
        cfg_pretrained = (argc > 7) ? argv[7]: 0; // pretrained_cfg        
        filename = (argc > 8) ? argv[8]: 0;
    }
    else     // Valid/Predict/Test/Video_demo arguments
    {
        solver = argv[3];   // Solver
        cfg_valid = argv[4];      // test_cfg
        weights = argv[5];  // pretrained_weights
    }

    if(0==strcmp(argv[2], "train")){
        multiloss_cost = 1;
        if(weights)
            train_multiloss_detector_pretrained(solver, cfg, cfg_valid, weights, cfg_pretrained, gpus, ngpus, clear);
        else
            train_multiloss_detector(solver, cfg, cfg_valid, weights, gpus, ngpus, clear);
    }

}


