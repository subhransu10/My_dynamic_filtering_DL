# mos/calibrate_temp.py
import torch, argparse
from torch.optim import LBFGS

def fit_temperature(logits, labels):
    # logits: [N], labels: [N] in {0,1}, on CPU tensors
    T = torch.ones(1, requires_grad=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    opt = LBFGS([T], lr=0.01, max_iter=100)

    logits = logits.detach()
    labels = labels.float().detach()

    def closure():
        opt.zero_grad()
        loss = criterion(logits / T.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-3))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_logits", required=True)  # torch.save({'logits':..., 'labels':...})
    args = ap.parse_args()
    payload = torch.load(args.val_logits, map_location="cpu")
    T = fit_temperature(payload["logits"], payload["labels"])
    print(f"Best temperature: {T:.4f}")
