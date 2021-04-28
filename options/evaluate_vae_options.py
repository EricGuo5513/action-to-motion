from options.base_vae_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Epoch which is loaded for evaluation')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/vae/", help='Save path of animation results')

        # Either replic_times or do_random is activated at one time
        self.parser.add_argument('--replic_times', type=int, default=1, help='Replication times of all categories')

        self.parser.add_argument('--do_random', action='store_true', help='Random generation')
        self.parser.add_argument('--num_samples', type=int, default=100, help='Number of generated')

        self.parser.add_argument('--batch_size', type=int, default=20, help='Batch size of training process')

        # Extension for name of saving files
        self.parser.add_argument('--name_ext', type=str, default="", help='Extension of save path')

        self.isTrain = False