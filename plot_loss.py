
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 20, 17

# read data
snr = 15
signal_train_losses = np.load('loss/signal_train_losses.npy')
signal_val_losses = np.load('loss/signal_val_losses.npy')
interf_train_losses = np.load('loss/interf_train_losses.npy')
interf_val_losses = np.load('loss/interf_val_losses.npy')

rate_train_losses = np.load(f'loss/rate_train_losses_{snr}dB.npy')
rate_val_losses = np.load(f'loss/rate_val_losses_{snr}dB.npy')

def plot_result(train_losses, val_losses, name):
    plt.figure(figsize=(6.5, 5))
    plt.plot(train_losses, label='Training loss', linewidth=2.0)
    plt.plot(val_losses, label='Validation loss', linewidth=2.0)
    plt.xlabel('Epochs', fontsize=font1)
    plt.ylabel('Loss', fontsize=font1)
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.legend(fontsize=font2, handlelength=1)
    # plt.yscale('log')

    if name == "signal":
        plt.ylim([0.05, 0.60])
        # plt.yticks(np.arange(0.1, 0.51, 0.1))
        plt.xticks(range(0,len(train_losses)+100, 200), fontsize=font1)
    elif name == "interf":
        plt.ylim([0.25, 1.00])
        plt.xticks(range(0, len(train_losses)+20, 50), fontsize=font1)
    else:
        plt.xlim([-5,104])
        plt.ylim([0.21, 0.28])
        plt.xticks(range(0, len(train_losses)+20, 20), fontsize=font1)

    plt.tight_layout()
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)

plot_result(signal_train_losses, signal_val_losses, "signal")
plt.savefig('result/signal_loss.pdf')
plot_result(interf_train_losses, interf_val_losses, "interf")
plt.savefig('result/interf_loss.pdf')
plot_result(rate_train_losses, rate_val_losses, "rate")
plt.savefig('result/rate_loss.pdf')
plt.show()


# plt.figure(figsize=(8, 6))
# plt.plot(signal_train_losses, label='Signal training loss',
#          linestyle = '-', linewidth=2.0)
# plt.plot(signal_val_losses, label='Signal validation loss',
#          linestyle = '--', linewidth=2.0)
# plt.plot(interf_train_losses, label='Interf training loss',
#          linestyle = '-', linewidth=2.0)
# plt.plot(interf_val_losses, label='Interf validation loss',
#          linestyle = '--', linewidth=2.0)

# plt.xlabel('Epochs', fontsize=font1)
# plt.ylabel('Loss', fontsize=font1)
# # plt.yscale('log')
# plt.ylim([0.05, 0.80])
# plt.xticks(fontsize=font1)
# plt.yticks(fontsize=font1)
# plt.legend(fontsize=font1)
# plt.tight_layout()
# plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
# plt.show()