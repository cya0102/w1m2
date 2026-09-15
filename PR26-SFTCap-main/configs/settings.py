import os
import datetime
import argparse

__all__ = ['TotalConfigs', 'get_settings']


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')


def _settings():
    parser = argparse.ArgumentParser()
    """
    =========================General Settings===========================
    """
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--sample_numb', type=int, default=15,
                        help='how many frames would you like to sample from a given video')
    parser.add_argument('--model_name', type=str, default='Track_VC', help='which model you would like to train/test?')
    parser.add_argument('--clip_name', type=str, default='clip_b16', help='which model you would like to train/test?')
    parser.add_argument('--loglevel', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--SFT', action='store_true')
    parser.add_argument('--Only_T_or_S', type=str, default='None',
                        choices=['T', 'S', 'None'])

    parser.add_argument('--use_subject', action='store_true')
    parser.add_argument('--use_predict', action='store_true')
    parser.add_argument('--use_sentence', action='store_true')
    parser.add_argument('--use_module', type=str, default='transformer',
                        choices=['transformer', 'lstm', 'None'])
    # parser.add_argument('--use_transformer', action='store_true')
    # parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--use_ham', action='store_true')
    parser.add_argument('--sft_num', type=int, default=3)
    """
    =========================Sys Settings===========================
    """
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--init_method', default="tcp://localhost:2222")
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--multiprocess', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

    """
    =========================Data Settings===========================
    """
    parser.add_argument('--dataset', type=str, default='msvd')
    '''使用预处理/端到端'''
    parser.add_argument('--train_type', type=str, default='preprocess')
    '''预处理'''
    parser.add_argument('--data_root', type=str,
                        default='E:\SFT4Caps\data', help='all mavd and msrvtt')
    '''端到端（还未写）'''
    parser.add_argument('--checkpoints_dir', type=str, default='./output/checkpoints')
    parser.add_argument('--result_dir', type=str, default='-1')
    parser.add_argument('--log_dir', type=str, default='./log/experiment')
    parser.add_argument('--log_freq', type=int, default=1)
    """
    =========================Encoder Settings===========================
    """
    parser.add_argument('--visual_dim', type=int, default=512, help='dimention for inceptionresnetv2=2048,clip=512')
    parser.add_argument('--object_dim', type=int, default=2048, help='dimention for vg_objects')
    parser.add_argument('--semantics_dim', type=int, default=768, help='semantics embedding dim')
    parser.add_argument('--max_objects', type=int, default=8)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--entity_encoder_layer', type=int, default=2)
    parser.add_argument('--entity_decoder_layer', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--transformer_activation', type=str, default='relu')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--tracker_encoder_layer', type=int, default=2)
    """
    =========================Word Dict Settings===========================
    """
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--sos_idx', type=int, default=1)
    parser.add_argument('--unk_idx', type=int, default=2)
    parser.add_argument('--n_vocab', type=int, default=-1, help='how many different words are there in the dataset')
    """
    =========================Decoder Settings===========================
    """
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    """
    =========================Training Settings===========================
    """
    parser.add_argument('--T_loss', type=float, default=8.0)
    parser.add_argument('--lam_o', type=float, default=0.0)
    parser.add_argument('--lam_a', type=float, default=0.0)
    parser.add_argument('--distill_o', type=float, default=0.0)
    parser.add_argument('--distill_a', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lambda_entity', type=float, default=0.6)
    parser.add_argument('--lambda_predicate', type=float, default=0.3)
    parser.add_argument('--lambda_sentence', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--save_checkpoints_every', type=int, default=100)
    parser.add_argument('--save_checkpoints_path', type=str, default='-1')
    parser.add_argument('--save_freq', type=int, default=1)

    """
    =========================Testing Settings===========================
    """
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_caption_len', type=int, default=22)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--result_path', type=str, default='-1')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--schedule', type=str, default="warmup_constant")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--clip_lr', type=float, default=2e-6)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.95)
    args = parser.parse_args()
    return args


class TotalConfigs:
    def __init__(self, args):
        self.data = DataConfigs(args)
        self.dict = DictConfigs(args)
        self.encoder = EncoderConfigs(args)
        self.decoder = DecoderConfigs(args)
        self.train = TrainingConfigs(args)
        self.test = TestConfigs(args)
        self.bert = BertConfigs(args)

        self.seed = args.seed
        self.bsz = args.bsz
        self.drop_prob = args.drop_prob
        self.model_name = args.model_name
        self.sample_numb = args.sample_numb
        self.loglevel = args.loglevel
        self.SFT = args.SFT
        self.use_subject = args.use_subject
        self.use_predict = args.use_predict
        self.use_sentence = args.use_sentence
        # self.use_transformer = args.use_transformer
        # self.use_lstm = args.use_lstm
        self.use_ham = args.use_ham
        self.use_module = args.use_module
        if self.SFT:
            self.sft_num = args.sft_num
        else:
            self.Only_T_or_S = args.Only_T_or_S
            assert self.Only_T_or_S != 'None'
            if self.Only_T_or_S =="S":
                self.use_module = "None"
        self.num_gpu = args.num_gpu
        self.num_shards = args.num_shards
        self.shard_id = args.shard_id
        self.init_method = args.init_method
        self.distributed = args.distributed
        self.multiprocess = args.multiprocess
        self.num_workers = args.num_workers
        self.local_rank = args.local_rank


class DataConfigs:
    def __init__(self, args):
        self.train_type = args.train_type
        self.dataset = args.dataset
        self.data_root = args.data_root
        self.checkpoints_dir = args.checkpoints_dir
        '''预处理'''
        # data root
        self.data_root = os.path.join(self.data_root, self.dataset)
        self.visual_dir = os.path.join(self.data_root, 'visual')
        if args.clip_name == "clip_b16":
            self.clip_weights = "E:\SFT4Caps\model_zoo\clip_model\ViT-B-16.pt"
        elif args.clip_name == "clip_l14":
            self.clip_weights = "E:\SFT4Caps\model_zoo\clip_model\ViT-L-14.pt"
        elif args.clip_name == "clip_b32":
            self.clip_weights = "E:\SFT4Caps\model_zoo\clip_model\ViT-B-32.pt"
        # lang root
        self.language_dir = os.path.join(self.data_root, 'language')
        self.word2idx_path = os.path.join(self.language_dir, 'word2idx.pkl')
        self.idx2word_path = os.path.join(self.language_dir, 'idx2word.pkl')
        self.embedding_weights_path = os.path.join(self.language_dir, 'embedding_weights.pkl')
        self.vid2groundtruth_path = os.path.join(self.language_dir, 'vid2groundtruth.pkl')
        self.visual_features = os.path.join(self.visual_dir,
                                            '{clip_name}/frame_feature'.format(clip_name=args.clip_name))
        self.object_features = os.path.join(self.visual_dir, '{}_vg_objects_{}.hdf5'.format(args.dataset, '{}'))
        if self.dataset == 'msvd':
            self.vid2language_features = os.path.join(self.language_dir, 'vid2language.pkl')
        elif self.dataset == 'msrvtt':
            self.vid2language_features = os.path.join(self.language_dir, 'vid2language_{}.pkl'.format('{}'))
        self.ann_root = os.path.join(self.language_dir, '{dataset}_caption.json'.format(dataset=args.dataset))

        # dataset split part
        self.videos_split = os.path.join(self.data_root, 'splits/{}_{}_list.pkl'.format(self.dataset, '{}'))

        '''log set'''
        self.log_dir = args.log_dir
        self.logger_file = 'logger.log'
        self.log_freq = args.log_freq


class DictConfigs:
    def __init__(self, args):
        self.eos_idx = args.eos_idx
        self.sos_idx = args.sos_idx
        self.unk_idx = args.unk_idx
        self.n_vocab = args.n_vocab


class BertConfigs:
    def __init__(self, args):
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.lr = args.lr
        self.warmup = args.warmup
        self.schedule = args.schedule
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.clip_lr = args.clip_lr
        self.lr_decay_gamma = args.lr_decay_gamma


class EncoderConfigs:
    def __init__(self, args):
        if args.clip_name == 'clip_b16' or args.clip_name == 'clip_b32':
            self.visual_dim = 512
        else:
            self.visual_dim = 768
        # self.visual_dim = args.visual_dim
        self.object_dim = args.object_dim
        self.semantics_dim = args.semantics_dim
        self.word_dim = args.word_embedding_dim
        self.max_objects = args.max_objects
        # transformer setting
        self.nheads = args.nheads
        self.entity_encoder_layer = args.entity_encoder_layer
        self.entity_decoder_layer = args.entity_decoder_layer
        self.dim_feedforward = args.dim_feedforward
        self.transformer_activation = args.transformer_activation
        self.d_model = args.d_model
        self.trans_dropout = args.transformer_dropout
        # tracker_encoder setting
        self.tracker_encoder_layer = args.tracker_encoder_layer


class DecoderConfigs:
    def __init__(self, args):
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.n_vocab = -1


class TrainingConfigs:
    def __init__(self, args):
        self.grad_clip = args.grad_clip
        self.learning_rate = args.learning_rate
        self.lambda_entity = args.lambda_entity
        self.lambda_predicate = args.lambda_predicate
        self.lambda_sentence = args.lambda_sentence
        self.max_epochs = args.max_epochs
        self.T_loss = args.T_loss * 0.1
        assert 1.0 >= self.T_loss >= 0.0
        self.lam_o = args.lam_o
        self.distill_o = args.distill_o
        self.lam_a = args.lam_a
        self.distill_a = args.distill_a
        self.visualize_every = args.visualize_every
        # self.checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset_name)
        self.checkpoints_dir = os.path.join(args.checkpoints_dir,
                                            "{}/SFT_{}_checkpoint_{}_Tloss_{}_KLo_{}_KLa_{}_DLo_{}_DLa_{}".format(args.dataset, args.SFT,
                                                                                      get_timestamp(), self.T_loss, self.lam_o, self.lam_a, self.distill_o, self.distill_a))
        self.save_checkpoints_every = args.save_checkpoints_every
        if args.SFT:
            self.save_checkpoints_path = os.path.join(self.checkpoints_dir,
                                                      '{clip_name}_epochs_{max_epochs}_lr_{lr}_use_{trans}_module_ham_{ham}_max_objects_{mo}.ckpt'.format(
                                                          clip_name=args.clip_name,
                                                          max_epochs=args.max_epochs,
                                                          lr=self.learning_rate,
                                                          trans=args.use_module,
                                                          ham=args.use_ham,
                                                          mo=args.max_objects))
        else:
            self.save_checkpoints_path = os.path.join(self.checkpoints_dir,
                                                      '{clip_name}_epochs_{max_epochs}_lr_{lr}_useonly_{TorS}_use_{trans}_module_ham_{ham}_max_objects_{mo}.ckpt'.format(
                                                          clip_name=args.clip_name,
                                                          max_epochs=args.max_epochs,
                                                          lr=self.learning_rate,
                                                          TorS=args.Only_T_or_S,
                                                          trans=args.use_module,
                                                          ham=args.use_ham,
                                                          mo=args.max_objects))
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if args.save_checkpoints_path != '-1':
            self.save_checkpoints_path = args.save_checkpoints_path
        self.evaluate_dir = os.path.join(self.checkpoints_dir, 'evaluate.txt')
        self.captions_dir = os.path.join(self.checkpoints_dir, 'gen_captions.txt')
        self.save_freq = args.save_freq


class TestConfigs:
    def __init__(self, args):
        self.beam_size = args.beam_size
        self.max_caption_len = args.max_caption_len
        self.temperature = args.temperature
        self.result_dir = os.path.join('./results/SFT_{}_{dataset_name}'.format(args.SFT, dataset_name=args.dataset))
        if args.result_dir != '-1':
            self.result_dir = args.result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_path = os.path.join(
            self.result_dir,
            '{clip_name}_epochs_{max_epochs}_lr_{lr}_useonly_{TorS}_use_{trans}_module_ham_{ham}_max_objects_{mo}.pkl'.format(
                clip_name=args.clip_name,
                max_epochs=args.max_epochs,
                lr=args.learning_rate,
                TorS=args.Only_T_or_S,
                trans=args.use_module,
                ham=args.use_ham,
                mo=args.max_objects)
        )
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if args.result_path != '-1':
            self.result_path = args.result_path


def get_settings():
    args = _settings()
    configs = TotalConfigs(args=args)
    return configs
