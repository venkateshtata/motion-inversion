import os.path as osp
import os
import functools
from tqdm import tqdm
import torch
import sys

from utils.visualization import motion2bvh_rot
from utils.pre_run import OptimOptions, load_all_form_checkpoint
from motion_class import DynamicData

def inverse_optim(args, g_ema, discriminator, device, mean_latent, target_motion, motion_statics, normalisation_data):
    from models.inverse_losses import DiscriminatorLoss, LatentCenterRegularizer, PositionLoss
    
    pos_loss_local = PositionLoss(motion_statics, normalisation_data, device, True, args.foot, args.use_velocity, local_frame=args.use_local_pos)

    target_motion = torch.tensor(target_motion, device=device, dtype=torch.float32)
    target_motion = target_motion.permute(0, 2, 1, 3)
    

    # Convert normalization values to PyTorch tensors and move them to the correct device
    mean_tensor = torch.tensor(normalisation_data["mean"], dtype=torch.float32, device=device)
    std_tensor = torch.tensor(normalisation_data["std"], dtype=torch.float32, device=device)

    # Normalize target motion
    target_motion_data = DynamicData(target_motion, motion_statics, use_velocity=args.use_velocity)
    target_motion_data = target_motion_data.un_normalise(mean_tensor, std_tensor)

    criteria = eval(args.criteria)(args)

    if args.lambda_disc > 0:
        disc_criteria = DiscriminatorLoss(args, discriminator)
    else:
        disc_criteria = None

    if args.lambda_latent_center > 0:
        latent_center_criteria = LatentCenterRegularizer(args, mean_latent)
    else:
        latent_center_criteria = None
    args.n_iters = 10000

    loop = tqdm(range(args.n_iters), desc='Sampling')
    if args.Wplus:
        n_latent = g_ema.n_latent
        target_W = torch.randn(1, n_latent, args.latent, device=device, requires_grad=True)
        print('target_W shape: ', target_W.shape)
    else:
        target_W = torch.randn(1, args.latent, device=device, requires_grad=True)
    optim = torch.optim.Adam([target_W], lr=args.lr)

    os.makedirs(args.out_path, exist_ok=True)

    save_bvh = functools.partial(motion2bvh_rot)

    save_bvh(target_motion_data, osp.join(args.out_path, 'target.bvh'))

    for i in loop:
        motion, _, _ = g_ema([target_W], truncation=args.truncation, truncation_latent=mean_latent,
                             input_is_latent=True)

       # Normalize predicted motion
        motion_data = DynamicData(motion, motion_statics, use_velocity=args.use_velocity)

        loss = loss_main = criteria(motion, target_motion)
        if disc_criteria is not None:
            disc_loss = disc_criteria(motion)
            loss += args.lambda_disc * disc_loss
        else:
            disc_loss = torch.tensor(0.)
        if latent_center_criteria is not None:
            reg_loss = latent_center_criteria(target_W)
            loss += args.lambda_latent_center * reg_loss
        else:
            reg_loss = torch.tensor(0.)

        pos_loss = pos_loss_local(motion, target_motion)
        loss += args.lambda_pos * pos_loss

        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        description_str = 'loss: {:.4f}, disc_loss: {:.4f}, reg_loss: {:.4f}, pos_loss: {:.4f}'.format(loss_main.item(), disc_loss.item(), reg_loss.item(), pos_loss.item())
        loop.set_description(description_str)

        if (i + 1) % 500 == 0:
            motion_data = motion_data.un_normalise(mean_tensor, std_tensor)
            save_bvh(motion_data, osp.join(args.out_path, '{}_inverse_optim.bvh'.format(i + 1)))
            torch.save({'W': target_W}, osp.join(args.out_path, '{}_inverse_optim.pth'.format(i + 1)))

    return target_W.detach().cpu().numpy(), motion.detach().cpu().numpy()

def main(args_not_parsed):
    parser = OptimOptions()
    args = parser.parse_args(args_not_parsed)

    g_ema, discriminator, motion_data, mean_latent, motion_statics, normalisation_data, args = load_all_form_checkpoint(args.ckpt, args, return_motion_data=True)


    target_motion = motion_data[[args.target_idx]]
    res = inverse_optim(args, g_ema, discriminator, args.device, mean_latent, target_motion, motion_statics, normalisation_data)

if __name__ == "__main__":
    main(sys.argv[1:])