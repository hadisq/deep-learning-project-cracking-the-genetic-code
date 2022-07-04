import sys
import numpy as np
import pandas as pd
import math
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
#mpl.use('qt4agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.axes3d import Axes3D
#from uuv_trajectory_generator import BezierCurve
from scipy import interpolate
from scipy.interpolate import splprep, splev
from scipy import linalg
from scipy import integrate
import mpmath
import splines
import sympy
from matplotlib.animation import PillowWriter
# initiate the lists required for the plots:
epoch_01 = list()
tra_loss_01 = list()
tst_loss_01 = list()
tra_accuracy_01 = list()
tst_accuracy_01 = list()

epoch_02 =list()
tra_loss_02 = list()
tst_loss_02 = list()
tra_accuracy_02 = list()
tst_accuracy_02 = list()

epoch_03 =list()
tra_loss_03 = list()
tst_loss_03 = list()
tra_accuracy_03 = list()
tst_accuracy_03 = list()

epoch_04 = list()
tra_loss_04 = list()
tst_loss_04 = list()
tra_accuracy_04 = list()
tst_accuracy_04 = list()

epoch_05 = list()
tra_loss_05 = list()
tst_loss_05 = list()
tra_accuracy_05 = list()
tst_accuracy_05 = list()

epoch_06 = list()
tra_loss_06 = list()
tst_loss_06 = list()
tra_accuracy_06 = list()
tst_accuracy_06 = list()

# read the files to retrieve the scoring metrics from:
#-----------------------------------------------------

def loadFile(path, fileName):
    # initiate the lists required for the plots:
    iteration = list()
    traLoss = list()
    tstLoss = list()
    traAccu = list()
    tstAccu = list()
    infile = open(path+fileName, 'r')
    header1 = infile.readline()
    header2 = infile.readline()
    line = infile.readline()
    lineCount = 0
    while line != '':
        lineCount += 1
        lineString = line.rstrip('\n') # this removes (strips) the '\n' newline character from the line read.
        columnList = lineString.split('\t')  # this reads each column in the string by splitting between tab delimeters.
        iteration.append(int(columnList[0]))
        traLoss.append(float(columnList[1]))
        tstLoss.append(float(columnList[2]))
        traAccu.append(float(columnList[3]))
        tstAccu.append(float(columnList[4]))
        line = infile.readline()
    infile.close()
    print('number of epochs read = ', lineCount)
    return iteration, traLoss, tstLoss, traAccu, tstAccu

# select the correct files to retrieve the data from:
filePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/"
#fileToRead_01 = "MLP_40_iterations64bitsNOWeights.txt"
#fileToRead_02 = "MLPembedding64to2tanh_40epFreqFalse.txt"
#fileToRead_03 = "MLPembedding64to10tanh_40epFreqFalse.txt"

#fileToRead_01 = "MLP_40_iterations64bitsNOWeights.txt"
#fileToRead_02 = "MLP_40_iterationsMLP64bits_weights.txt"
#fileToRead_03 = "MLP40epMLP12bits.txt"
#fileToRead_04 = "MLP40epMLP12bits_weights.txt"

#fileToRead_01 = "MLP64bits.txt" #"MLP_40_iterations64bitsNOWeights.txt"
#fileToRead_02 = "MLP64bits_weights.txt"# "RNN2layers64_40ep.txt"
#fileToRead_03 =  "MLP40epMLP12bits.txt" #"GRU2layers64_40ep.txt"
#fileToRead_04 =  "MLP12bitsOHEweights.txt"#"LSTM64_40ep.txt"

fileToRead_01 = "MLP64bits.txt" #"MLP_40_iterations64bitsNOWeights.txt"
fileToRead_02 = "RNN2layers64_40ep.txt"# "RNN2layers64_40ep.txt"
fileToRead_03 =  "GRU2layers64_40ep.txt" #"GRU2layers64_40ep.txt"
fileToRead_04 =  "LSTM64_40ep.txt"#"LSTM64_40ep.txt"
fileToRead_05 =  "MLP64bitsLayer1024.txt"#"LSTM64_40ep.txt"
fileToRead_06 =  "RNN2layers256_40ep.txt"#"LSTM64_40ep.txt"

#fileToRead_01 = "MLP64bits.txt" #"MLP_40_iterations64bitsNOWeights.txt"
#fileToRead_02 = "MLPembedding64to2tanh_40epFreqFalse.txt"# "no weights adjustement"
#fileToRead_03 = "MLPembedding64to2tanh_40epFreqTrue.txt"# "weights adjustement"
#fileToRead_04 =  "MLPembedding64to10tanh_40epFreqFalse.txt" #"noweights adjustment"
#fileToRead_05 =  "MLPembedding64to10tanh_40epfreqTrue.txt"#"weight adjustment"

epoch_01, tra_loss_01, tst_loss_01, tra_accuracy_01, tst_accuracy_01 = loadFile(filePATH, fileToRead_01)
print('loss', tra_loss_01)
print('accu', tra_accuracy_01)

epoch_02, tra_loss_02, tst_loss_02, tra_accuracy_02, tst_accuracy_02 = loadFile(filePATH, fileToRead_02)
print('loss', tra_loss_02)
print('accu', tra_accuracy_02)

epoch_03, tra_loss_03, tst_loss_03, tra_accuracy_03, tst_accuracy_03 = loadFile(filePATH, fileToRead_03)
print('loss', tra_loss_03)
print('accu', tra_accuracy_03)

epoch_04, tra_loss_04, tst_loss_04, tra_accuracy_04, tst_accuracy_04 = loadFile(filePATH, fileToRead_04)
print('loss', tra_loss_04)
print('accu', tra_accuracy_04)

epoch_05, tra_loss_05, tst_loss_05, tra_accuracy_05, tst_accuracy_05 = loadFile(filePATH, fileToRead_05)
print('loss', tra_loss_05)
print('accu', tra_accuracy_05)

epoch_06, tra_loss_06, tst_loss_06, tra_accuracy_06, tst_accuracy_06 = loadFile(filePATH, fileToRead_06)
print('loss', tra_loss_05)
print('accu', tra_accuracy_05)

#sys.exit()
# normalize the loss relative to maxLoss value:
def scaleLoss(lossList):
    maxLoss = max(lossList)
    scaledLoss = list()
    for i in range(len(lossList)):
        if maxLoss >0:
            scaledLoss.append(lossList[i]/maxLoss)
        else:
            scaledLoss.append(lossList[i] / (maxLoss+0.000001))
    return scaledLoss

# Plot the graph of the potential and the graph of axial electric field or axial force:
plt.clf()
fig, ax = plt.subplots(4, 1, figsize=(8, 16))
epSup = 15 # upper bound for epochs
ax[0].plot(epoch_01[:epSup], tra_accuracy_01[:epSup], lw=2.0, ls="--", color="lightgray", label="MLP64 linear fc")
ax[0].plot(epoch_01[:epSup], tra_accuracy_05[:epSup], lw=2.0, ls=":", color="red", label="MLP64x1024 linear fc")
ax[0].plot(epoch_01[:epSup], tra_accuracy_02[:epSup], lw=2.0, ls="-", color="lightgray", alpha=0.90, label="RNN 2 layers OHE 3 x 4 bits hidden size 64")
ax[0].plot(epoch_01[:epSup], tra_accuracy_06[:epSup], lw=2.0, ls="-", color="black", alpha=0.70, label="RNN 2 layers OHE 3 x 4 bits hidden size 256")
ax[0].plot(epoch_01[:epSup], tra_accuracy_03[:epSup], lw=2.0, ls="-.", color="gray", label="GRU 2 layers OHE 3 x 4 bits")
ax[0].plot(epoch_01[:epSup], tra_accuracy_04[:epSup], lw=2.0, ls=":", color="gray", label="LSTM 2 layers OHE 3 x 4 bits")
#ax[0].plot(epoch_01[:epSup], tra_accuracy_05[:epSup], lw=2.0, ls="-.", color="black", label="MLP64 embedding d=10 adjusting for weights")
#ax[0].plot(epoch_01[:epSup], tra_accuracy_03[:epSup], marker="+", color="gray", label="MLP12 linear fc")
#ax[0].plot(zTableStrucData[:], PhiOfZtot[:], lw=2.0, ls="-", color="gray", label="Electrostatic potential (mV)")
ax[0].set_ylim(0, 1.05)
ax[0].set_xlabel(r"epochs", fontsize=20)
ax[0].set_ylabel(r"Training accuracy", fontsize=20)
ax[0].legend()
#vertical line for tunnel entry port:
#ax[0].vlines(x=-8.75, ymin=min(PhiOfZphosphoBIS)*1.05, ymax=0.0, color='darkgray', linestyles='--', lw=1.0)
#vertical line at z coordinate of A2485 P loop P site at PTC:
#ax[0].vlines(x=(16.01294-zShift)/10.0, ymin=min(PhiOfZphosphoBIS)*1.05, ymax=0.0, color='darkgray', linestyles='--', lw=1.0)

ax[1].plot(epoch_01[2:epSup], tst_accuracy_01[2:epSup], lw=2.0, ls="--", color="lightgray", alpha=0.70, label="MLP64 linear fc")
ax[1].plot(epoch_01[:epSup], tst_accuracy_05[:epSup], lw=2.0, ls=":", color="red", label="MLP64x1024 linear fc")
ax[1].plot(epoch_01[2:epSup], tst_accuracy_02[2:epSup], lw=2.0, ls="-", color="lightgray", alpha=0.90, label="RNN 2 layers OHE 3 x 4 bits hidden size 64")
ax[1].plot(epoch_01[:epSup], tst_accuracy_06[:epSup], lw=2.0, ls="-", color="black", alpha=0.50, label="RNN 2 layers OHE 3 x 4 bits hidden size 256")
ax[1].plot(epoch_01[:epSup], tst_accuracy_03[:epSup], lw=2.0, ls="-.", color="gray", label="GRU 2 layers OHE 3 x 4 bits")
ax[1].plot(epoch_01[:epSup], tst_accuracy_04[:epSup], lw=2.0, ls=":", color="gray", label="LSTM 2 layers OHE 3 x 4 bits")
#ax[1].plot(epoch_01[:epSup], tst_accuracy_05[:epSup], lw=2.0, ls="-.", color="black", label="MLP64 embedding d=10 adjusting for weights")

#ax[0].plot(zTableStrucData[:], PhiOfZtot[:], lw=2.0, ls="-", color="gray", label="Electrostatic potential (mV)")
ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel(r"epochs", fontsize=20)
ax[1].set_ylabel(r"Test accuracy", fontsize=20)
ax[1].legend(loc='lower right')

ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_01)[:epSup], lw=2.0, ls="--", color="darkblue", label="MLP64 linear fc")
ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_05)[:epSup], lw=2.0, ls=":", color="red", label="MLP64x1024 linear fc")
ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_02)[:epSup], lw=2.0, ls="-", color="lightblue", alpha=0.6, label="RNN 2 layers OHE 3 x 4 bits hidden size 64")
ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_06)[:epSup], lw=2.0, ls="--", color="lightgray", alpha=0.7,label="RNN 2 layers OHE 3 x 4 bits hidden size 256")
ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_03)[:epSup], lw=2.0, ls="-.", color="darkblue", label="GRU 2 layers OHE 3 x 4 bits")
ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_04)[:epSup], lw=2.0, ls=":", color="blue", label="LSTM 2 layers OHE 3 x 4 bits")
#ax[2].plot(epoch_01[:epSup], scaleLoss(tra_loss_05)[:epSup], lw=2.0, ls="-.", color="blue", label="MLP64 embedding d=10 adjusting for weights")

ax[2].set_ylim(0, 1.05)
ax[2].set_xlabel(r"epochs", fontsize=20)
ax[2].set_ylabel(r"Training Loss"+'\n'+r"(scaled to max)", fontsize=18)
ax[2].legend()
#vertical line for tunnel entry port:
#ax[1].vlines(x=-8.75, ymin=0.0, ymax=max(PhiOfZResBIS)*1.05, color='darkgray', linestyles='--', lw=1.0)
#vertical line at z coordinate of A2485 P loop P site at PTC:
#ax[1].vlines(x=(16.01294-zShift)/10.0, ymin=0.0, ymax=max(PhiOfZResBIS)*1.05, color='darkgray', linestyles='--', lw=1.0)


#ax[2].plot(epoch_01[:], tra_loss_01[:], lw=2.0, ls="-", color="black", label="Electrostatic potential (mV)")
#ax[2].plot(zTableStrucData[:], PhiOfZtotBIS[:], lw=2.0, ls="-", color="orange", label="Electrostatic potential (mV)")
#ax[2].plot(zTable[:], PhiOfZ2[:], lw=1.0, ls="--", color="black", label="Electrostatic potential (mV)")
#ax[2].plot(zTable[:], PhiOfZ3[:], lw=1.0, ls="--", color="grey", label="Electrostatic potential (mV)")
#ax[2].set_ylim(min(PhiOfZ)*1.05, 0)
#ax[2].set_xlabel(r"$z$ (nm)", fontsize=20)
#ax[2].set_ylabel(r"$\Phi(z)$ (mV)", fontsize=20)
#vertical line for tunnel entry port:
#ax[2].vlines(x=-8.75, ymin=min(PhiOfZ)*1.05, ymax=0.0, color='darkgray', linestyles='--', lw=1.0)
#vertical line at z coordinate of A2485 P loop P site at PTC:
#ax[2].vlines(x=(16.01294-zShift)/10.0, ymin=min(PhiOfZ)*1.05, ymax=0.0, color='darkgray', linestyles='--', lw=1.0)


ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_01)[:epSup], lw=2.0, ls="--", color="darkblue", label="MLP64 linear fc")
ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_05)[:epSup], lw=2.0, ls=":", color="red", label="MLP64x1024 linear fc")
ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_02)[:epSup], lw=2.0, ls="-", color="lightblue", alpha=0.60,label="RNN 2 layers OHE 3 x 4 bits hidden size 64")
ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_06)[:epSup], lw=2.0, ls="--", color="lightgray", alpha=0.70,label="RNN 2 layers OHE 3 x 4 bits hidden size 256")
ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_03)[:epSup], lw=2.0, ls="-.", color="darkblue", label="GRU 2 layers OHE 3 x 4 bits")
ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_04)[:epSup], lw=2.0, ls=":", color="blue", label="LSTM 2 layers OHE 3 x 4 bits")
#ax[3].plot(epoch_01[:epSup], scaleLoss(tst_loss_05)[:epSup], lw=2.0, ls="-.", color="blue", label="MLP64 embedding d=10 adjusting for weights")


ax[3].set_ylim(0, 1.05)
ax[3].set_xlabel(r"epochs", fontsize=20)
ax[3].set_ylabel(r"Test Loss"+'\n'+r"(scaled to max)", fontsize=18)
ax[3].legend()
#plt.show()
# save the plot:
plt.savefig(r"C:\Users\marcjoiret\Desktop\MARC\TISSUE ENGINEERING and REGENERATIVE MEDICINE\DEEP LEARNING\Project\FIG\figNNrnnComparedDEEP.svg")
plt.savefig(r"C:\Users\marcjoiret\Desktop\MARC\TISSUE ENGINEERING and REGENERATIVE MEDICINE\DEEP LEARNING\Project\FIG\figNNrnnComparedDEEP.pdf")
sys.exit()