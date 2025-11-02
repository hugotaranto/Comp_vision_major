import numpy as np
import depth_pro
import torch

def load_model(checkpoint_path=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.half, checkpoint_path=checkpoint_path)
    model.eval()

    return model, transform


def depth_predict(image:np.ndarray, f_px, model, transform):

    image_t = transform(image)

    print("predicting")
    torch.cuda.reset_peak_memory_stats()
    prediction = model.infer(image_t, f_px=f_px)
    depth_tensor = prediction['depth']

    # print(f"Peak VRAM allocated during inference: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    # print(f"VRAM used after inference: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    # print(f"VRAM reserved after inference: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

    depth = depth_tensor.detach().cpu().numpy()

    del image_t, depth_tensor, prediction
    torch.cuda.empty_cache()

    return depth

     
