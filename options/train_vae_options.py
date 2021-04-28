from options.base_vae_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=100, help='Batch size of training process')

        self.parser.add_argument('--arbitrary_len', action='store_true', help='Enable variable length (batch_size has to'
                                                                              ' be 1 and motion_len will be disabled)')


        self.parser.add_argument('--skip_prob', type=float, default=0, help='Probability of skip frame while collecting loss')
        self.parser.add_argument('--tf_ratio', type=float, default=0.6, help='Teacher force learning ratio')

        self.parser.add_argument('--lambda_kld', type=float, default=0.0001, help='Weight of KL Divergence')
        self.parser.add_argument('--lambda_align', type=float, default=0.5, help='Weight of align loss')

        self.parser.add_argument('--use_geo_loss', action='store_true', help='Compute Geodesic Loss(Only when lie_enforce is enabled)')
        self.parser.add_argument('--lambda_trajec', type=float, default=0.8, help='Calculate trajectory align loss(Only when lie_enforce is enabled)')

        self.parser.add_argument('--is_continue', action="store_true", help='Continue training of checkpoint models')
        self.parser.add_argument('--iters', type=int, default=20, help='Training iterations')

        self.parser.add_argument('--plot_every', type=int, default=500, help='Sample frequency of iterations while plotting loss curve')
        self.parser.add_argument("--save_every", type=int, default=500,
                            help='Frequency of saving intermediate models during training')
        self.parser.add_argument("--eval_every", type=int, default=500,
                                 help='Frequency of save intermediate samples during training')
        self.parser.add_argument("--save_latest", type=int, default=500,
                                 help='Frequency of saving latest models during training')
        self.parser.add_argument('--print_every', type=int, default=50, help='Frequency of printing training progress')
        self.isTrain = True
