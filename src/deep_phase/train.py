import torch
from deep_phase.models import modified_resnet


def train_network(net, training_set, test_set, log, path,
                  epochs=50, batch_size=128, criterion=torch.nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.001)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    best_loss = None
    log.write('epoch,train_loss,train_acc,test_loss,test_acc\n')

    for epoch in range(epochs):  # loop over the dataset multiple times
        line = ""
        running_loss = 0.0
        running_acc = 0.0

        net.train()
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, labels, sup_label]
            inputs, labels, sup_label = data
            outputs = net(inputs)

            # forward + backward + optimize
            if isinstance(outputs, tuple):  # predicting superclass too
                sub_outputs, sup_outputs = outputs
                _, preds = torch.max(sub_outputs, 1)
                sub_loss = criterion(sub_outputs, labels)
                alpha = 0.75  # fraction subclass importance
                loss = alpha * sub_loss + (1-alpha) * criterion(sup_outputs, sup_label)
            else:
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                sub_loss = loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += sub_loss.item()
            running_acc += (preds == labels).sum() / len(labels)
        line += (
            f"{epoch + 1},{running_loss / len(train_loader):.4f},"
            f"{running_acc/ len(train_loader):.3f},"
        )

        net.eval()
        running_loss = 0.0
        with torch.no_grad():
            running_acc = 0
            for data in test_loader:
                outputs = net(data[0])
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, preds = torch.max(outputs, 1)
                running_acc += (preds == data[1]).sum()
                loss = criterion(outputs, data[1])
                running_loss += loss.item()
        mean_loss = running_loss / len(test_loader)
        mean_acc = running_acc/ len(test_set)
        line += f"{mean_loss:.4f},{mean_acc:.3f}"

        log.write(line + '\n')
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            modified_resnet.save_network(net, path)
            line += ' <-'
            log.flush()
        print(line.replace(',', '\t'))
