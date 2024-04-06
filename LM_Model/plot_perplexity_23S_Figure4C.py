'''
Extract the `val loss` values from a nanoGPT-like log file. Exponentiate and plot the perplexity values.
For simplicity, read in all 4 log files from an RNA LLM trial and plot.
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

def extract_data_from_file(filename):
    steps = []
    val_losses = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('step 1000000:'):
                return steps, val_losses
            elif line.startswith('step'):
                parts = line.strip().split()
                step = int(parts[1].replace(':', ''))
                val_loss = float(parts[-1])
                steps.append(step)
                val_losses.append(val_loss)

    return steps, val_losses

# Function for a 5-point running average
def running_average(values, N=5):
    cumsum = np.cumsum(np.insert(values, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_results(filenames, model_names, output_filename):
    plt.figure(figsize=(10, 7))

    # Define a color palette suitable for color-blind readers. Assume 3, 4 or 6 lines to plot.
    #colors = ['#FFD700', '#005AB5', '#009E73', '#DC6900', '#FF69B4', '#D147BD']
    colors = ['#005AB5', '#009E73', '#DC6900']
    #colors = ['#DC6900', '#005AB5', '#009E73', '#D147BD']

    for i, (filename, model_name) in enumerate(zip(filenames, model_names)):
        # Note, I want to cut out steps 0-1 from the plot, as they are off-scale.
        steps, val_losses = extract_data_from_file(filename)
        perplexities = [np.exp(loss) for loss in val_losses]
        
        # Apply smoothing to perplexities
        smoothed_perplexities = running_average(perplexities)

        # Use modulo operator to cycle through colors if there are more models than colors
        color = colors[i % len(colors)]

        # Adjust depending on log files, to define both beginning and end regions.
        # Also account for smooth in indexing.
        plt.plot(steps[2:-2], smoothed_perplexities[0:], label=model_name, linewidth=3.5, color=color)
        #plt.plot(steps[100:818], perplexities[100:818], label=model_name, linewidth=3.5, color=color)

    # Scale the x-axis by 10000 and update the label
    def scale_x(value, pos):
        return f'{value/10000:.0f}'

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(scale_x))
    plt.xlabel('Iterations (10,000s)', fontsize = 26)

    plt.ylabel('Validation Perplexity', fontsize = 26)
    plt.legend(fontsize = 18)
    plt.title ('Model Pretrained on 231 RNA Families', fontsize = 26)
    plt.grid(False)
    # Set limits for y axis
    plt.ylim(ymax=4.0)
    plt.ylim(ymin=0.9)  

    # Setting y and x axis to whole numbers only
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set the font size for the axes tick labels
    ax.tick_params(axis='both', which='major', labelsize=22)

    # Increase the axis line width
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2.0)  # Increase for thicker spines

    # Increase tick width
    plt.gca().xaxis.set_tick_params(width=2.0)  # Increase for thicker x-axis ticks
    plt.gca().yaxis.set_tick_params(width=2.0)  # Increase for thicker y-axis ticks

    # Save the plot to a file
    plt.savefig(output_filename, format='pdf')  # Saves as a .pdf file.

# Specifics for generating a plot go here. Use ''' ''' to comment out this set and use another, if wanted.
filenames = ['train_231RNAs_triples_0_18_6_300_rot_flash.log']
model_names = ['triple-nt tokens, individual layer RoPE']
output_filename = 'RNA_tokens_perplexity_23S_train_231RNAs_rot_flash.pdf'

plot_results(filenames, model_names, output_filename)
