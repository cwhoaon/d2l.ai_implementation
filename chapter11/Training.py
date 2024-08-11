import torch
import torch.nn as nn

def train_run_epoch(model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for src, tgt, src_valid_len, label in train_loader:
        src, tgt, label, src_valid_len = src.to(device), tgt.to(device), label.to(device), src_valid_len.to(device)

        outputs = model(src, tgt, src_valid_len)

        output = outputs.reshape(-1, outputs.size()[2])
        label = label.flatten()
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss

@torch.no_grad()
def test_run_epoch(model, device, test_loader, loss_fn):
    model.eval()
    total_loss = 0

    for src, tgt, src_valid_len, label in test_loader:
        src, tgt, label, src_valid_len = src.to(device), tgt.to(device), label.to(device), src_valid_len.to(device)

        outputs = model(src, tgt, src_valid_len)

        output = outputs.reshape(-1, outputs.size()[2])
        label = label.flatten()
        loss = loss_fn(output, label)

        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    return avg_loss

def train(model, device, train_loader, test_loader, optimizer, loss_fn, num_epochs):
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs+1):

        train_loss = train_run_epoch(model, device, train_loader, optimizer, loss_fn)
        train_losses.append(train_loss)

        test_loss = test_run_epoch(model, device, test_loader, loss_fn)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Train Loss = {train_loss:.20f}")
        print(f"Test Loss = {test_loss:.20f}")
        print()

    return train_losses, test_losses