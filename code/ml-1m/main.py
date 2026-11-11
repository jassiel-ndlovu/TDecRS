import os
import time
import argparse
import numpy as np

import torch
import device
from device import get_device
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils

import random
# Seed is read from the environment because the seeding block has to run before
# argparse. DECRS_SEED unset reproduces the original fixed seed of 1 exactly.
random_seed = int(os.environ.get('DECRS_SEED', 1))
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
    type=str,
    default="ml_1m",
    help="dataset option: 'ml_1m'")
parser.add_argument("--num_groups",
    type=int,
    default=18,
    help="item group number in the dataset, must change for each dataset")
parser.add_argument("--num_user_features",
    type=int,
    default=5,
    help="user feature num in the dataset, must change for each dataset")
parser.add_argument("--model", 
    type=str,
    default="DecNFM",
    help="model option: DecFM, DecNFM, DecFMR, FM, NFM, FMconf. FM/NFM are the uncorrected "
         "baselines and are ranked on their raw scores (no inference blending). "
         "FMconf is the Tier 1.5 control: FM with the --dcr_tag confounder as "
         "plain side features, also ranked raw. DecFMR (DecRS-R) is DecFM with its "
         "static-prior mediator plus FMconf's --dcr_tag features, Eq. 11 blended.")
parser.add_argument("--loss_type", 
    type=str,
    default="log_loss",
    help="loss option: 'square_loss' or 'log_loss'")
parser.add_argument("--optimizer",
    type=str,
    default="Adagrad",
    help="optimizer option: 'Adagrad', 'Adam', 'SGD', 'Momentum'")
parser.add_argument("--data_path",
    type=str,
    default="../../data/",  
    help="load data path")
parser.add_argument("--model_path", 
    type=str,
    default="./models/",
    help="saved model path")
parser.add_argument("--activation_function",
    type=str,
    default="relu",
    help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr", 
    type=float, 
    default=0.05, 
    help="learning rate")
parser.add_argument("--alpha", 
    type=float, 
    default=0.1, 
    help="alpha in the inference strategy")
parser.add_argument("--dropout", 
    default='[0.5, 0.2]',  
    help="dropout rate for FM and MLP")
parser.add_argument("--batch_size", 
    type=int, 
    default=128, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=300, 
    help="training epochs")
parser.add_argument("--hidden_factor", 
    type=int,
    default=64, 
    help="predictive factors numbers in the model")
parser.add_argument("--layers", 
    default='[64]', 
    help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda", 
    type=float, 
    default=0.1, 
    help="regularizer for bilinear layers")
parser.add_argument("--topN", 
    default='[10, 20, 50, 100]',  
    help="the recommended item num")
parser.add_argument("--batch_norm", 
    type=int,
    default=1,   
    help="use batch_norm or not. option: {1, 0}")
parser.add_argument("--pre_train", 
    action='store_true', 
    default=False,
    help="whether use the pre-train or not")
parser.add_argument("--pre_train_model_path", 
    type=str,
    default="./models/",
    help="pre_trained model_path")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="5",
    help="gpu card ID")
parser.add_argument("--num_workers",
    type=int,
    default=4,
    help="DataLoader worker processes. Must be 0 on Windows (spawn re-imports main).")
parser.add_argument("--dcr_tag",
    type=str,
    default="",
    help="DCR confounder tag, e.g. g0.9_k0_static. Empty = static DecRS prior "
         "(legacy path, bit-identical to before DCR was added). Build tags with "
         "tools/build_dcr_priors.py.")
parser.add_argument("--eval_every",
    type=int,
    default=10,
    help="run ranking evaluation every N epochs")
args = parser.parse_args()
print("args:", args)
print("random_seed:", random_seed, flush=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
print(device.describe(), flush=True)


#############################  PREPARE DATASET #########################
start_time = time.time()
train_path = args.data_path + '{}/training_list.npy'.format(args.dataset)
valid_path = args.data_path + '{}/validation_dict.npy'.format(args.dataset)
test_path = args.data_path + '{}/testing_dict.npy'.format(args.dataset)
user_feature_path = args.data_path + '{}/user_feature_file.npy'.format(args.dataset)
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)
item_category_path = args.data_path + '{}/item_category.npy'.format(args.dataset)
# ml_1m blends with the authors' released FM scores. Other datasets use FM
# scores generated from our own FM checkpoint by tools/gen_scores.py.
FM_score_path = ('./inference/FM_user_item_scores.npy' if args.dataset == 'ml_1m'
                 else './inference/{}_FM_user_item_scores.npy'.format(args.dataset))
confounder_prior_path = args.data_path + '{}/confounder_prior.npy'.format(args.dataset)

user_feature, item_feature, num_features = data_utils.map_features(user_feature_path, item_feature_path)
### by default, the last features in item_feature are the group features. the group num is set in the args.
num_groups = args.num_groups
# \bar{d} in the paper
confounder_prior = np.load(confounder_prior_path)

# DCR: per-row and per-user confounders, both None unless --dcr_tag is given
train_conf, user_conf = None, None
if args.dcr_tag:
    dcr_dir = args.data_path + '{}_dcr/'.format(args.dataset)
    train_conf = np.load(dcr_dir + 'train_confounder_{}.npy'.format(args.dcr_tag))
    user_conf = np.load(dcr_dir + 'user_confounder_{}.npy'.format(args.dcr_tag),
                        allow_pickle=True).item()
    print('DCR confounders: {} rows, {} users, tag {}'.format(
        len(train_conf), len(user_conf), args.dcr_tag), flush=True)

train_dataset = data_utils.FMData(train_path, user_feature, item_feature, args.loss_type,
                                  confounder=train_conf)
valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)
item_category = data_utils.loadData(item_category_path)
# Only the Dec* models blend with FM scores (see below), so the baselines can
# train before any score file exists -- on a new dataset FM has to come first.
FM_score = (None if args.model in ('FM', 'NFM', 'FMconf')
            else data_utils.loadData(FM_score_path))

train_loader = data.DataLoader(train_dataset, drop_last=True,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
user_rank_feature, user_kl_score, all_item_features, all_item_feature_values = evaluate.pre_ranking(\
             user_feature, item_feature, item_category, num_groups, train_dataset.train_dict, valid_dict, test_dict)

print('data ready. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
##############################  CREATE MODEL ###########################
if args.pre_train:
    assert os.path.exists(args.pre_train_model_path), 'lack of FM model'
    assert args.model == 'NFM', 'only support NFM for now'
    FM_model = torch.load(args.pre_train_model_path, map_location=get_device(), weights_only=False)
else:
    FM_model = None

if args.model == 'DecFM':
    model = model.DecFM(num_features, num_groups, args.hidden_factor, 
                args.batch_norm, eval(args.dropout), args.num_user_features, confounder_prior)
elif args.model == 'DecFMR':
    model = model.DecFMR(num_features, num_groups, args.hidden_factor,
                args.batch_norm, eval(args.dropout), args.num_user_features, confounder_prior)
elif args.model == 'DecNFM':
    model = model.DecNFM(num_features, num_groups, args.hidden_factor, 
                args.activation_function, eval(args.layers), 
                args.batch_norm, eval(args.dropout), args.num_user_features, confounder_prior)
elif args.model == 'FM':
    model = model.FM(num_features, args.hidden_factor,
                args.batch_norm, eval(args.dropout))
elif args.model == 'NFM':
    model = model.NFM(num_features, args.hidden_factor,
                args.activation_function, eval(args.layers),
                args.batch_norm, eval(args.dropout), FM_model)
elif args.model == 'FMconf':
    model = model.FMconf(num_features, num_groups, args.hidden_factor,
                args.batch_norm, eval(args.dropout), confounder_prior)
else:
    raise Exception('model not implemented!')

# The inference strategy of Eq. 11 blends the deconfounded score with a
# conventional FM score. That only applies to the Dec* models; the baselines
# (including the FMconf control) are ranked on their own scores, so the blend
# is disabled for them. FMconf can be blended offline from its checkpoint.
if args.model in ('FM', 'NFM', 'FMconf'):
    FM_score = None
    
model.to(get_device())
if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(
        model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

if args.loss_type == 'square_loss':
    criterion = nn.MSELoss(reduction='sum')
elif args.loss_type == 'log_loss':
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
else:
    raise Exception('loss type not implemented!')

# writer = SummaryWriter() # for visualization

###############################  TRAINING ############################

count, best_recall, best_epoch = 0, -100, -1
best_test_result = []
for epoch in range(args.epochs):
    model.train() # Enable dropout and batch_norm
    start_time = time.time()
    train_loader.dataset.ng_sample()
    
    for batch in train_loader:
        conf = None
        if len(batch) == 4:
            features, feature_values, label, conf = batch
            conf = conf.to(get_device())
        else:
            features, feature_values, label = batch
        features = features.to(get_device())
        feature_values = feature_values.to(get_device())
        label = label.to(get_device())

        model.zero_grad()
        if conf is None:
            prediction = model(features, feature_values)
        else:
            prediction = model(features, feature_values, conf)
        loss = criterion(prediction, label)
        loss += args.lamda * model.embeddings.weight.norm()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1
    
    if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
        model.eval()
        train_RMSE = evaluate.RMSE(model, args.model, train_loader)
        valid_result, test_result, calibration_results = evaluate.Ranking(model, args.model, valid_dict, \
                                                      test_dict, train_dataset.train_dict, item_feature, \
                                                      user_rank_feature, user_kl_score, \
                                                      all_item_features, all_item_feature_values, \
                                                      FM_score, 4096, eval(args.topN), \
                                                      num_groups, args.alpha,
                                                      user_confounder=user_conf)
        print('---'*18)
        print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        evaluate.print_results(train_RMSE, valid_result, test_result, calibration_results)

        if valid_result[1][0] > best_recall: # recall@10 for selection
            best_recall, best_epoch = valid_result[1][0], epoch
            best_test_result = test_result
            print("------------Best model, saving...------------")
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                # The tag and seed must appear in the filename: without them every
                # cell of a grid writes the same path and silently overwrites the
                # previous cell's checkpoint.
                _suffix = ('_' + args.dcr_tag if args.dcr_tag else '') + '_seed' + str(random_seed)
                torch.save(model, '{}{}_{}_{}lr_{}bs_{}dropout_{}alpha_{}lamda{}.pth'.format(
                    args.model_path, args.model, args.dataset, args.lr, \
                    args.batch_size, args.dropout, args.alpha, args.lamda, _suffix))

print("End. Best epoch {:03d}".format(best_epoch))
evaluate.print_results(None, None, best_test_result, None)


