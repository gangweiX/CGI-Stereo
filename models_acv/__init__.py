from models_acv.CGF_ACV import ACVNet
from models_acv.loss import model_loss_train, model_loss_test, model_loss_train_attn_only, model_loss_train_freeze_attn

__models__ = {
    "acvnet": ACVNet
}
