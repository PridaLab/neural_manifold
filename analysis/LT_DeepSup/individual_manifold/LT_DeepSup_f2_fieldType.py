import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from datetime import datetime
from tkinter import *
from matplotlib.ticker import MaxNLocator
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

class windowClassifyPC:
    def __init__(self,window, PC, animalName = None, saveDir=None):
        self.window = window
        #save variables
        if saveDir:
            self.saveDir = saveDir
        else:
            self.saveDir = os.getcwd()
            print(f'Setting save dir to current directory: {self.saveDir}')
        if animalName:
            self.animalName = animalName
        else:
            self.animalName = 'Unknown'
            print(f'Setting animal name to: {self.animalName}')
        self.saveNameType = self.animalName+'_cellType_'+datetime.now().strftime('%d%m%y')+'.npy'
        self.saveNameCorr = self.animalName+'_cellCorrs'+datetime.now().strftime('%d%m%y')+'.npy'

        #place cells structure
        self.PC = PC
        self.numCells = PC['neu_sig'].shape[1]

        self.cellsPerRow = 3
        self.cellsPerColumn = 4
        self.cellsPerSlide = self.cellsPerRow*self.cellsPerColumn 
        self.numSlides = np.ceil(self.numCells/self.cellsPerSlide).astype(int)

        #title
        self.titleCell=Label(window, text=f"Classifying Place Cells", fg='black', font=("Arial", 20))
        self.titleCell.place(x=650, y=10)

        #previous cell slide buttom
        self.btnPre=Button(window, text="Previous", fg='black', command=self.previous_slide)
        self.btnPre.place(x=800, y=10)

        #next cell slide buttom
        self.btnNext=Button(window, text="Next", fg='black', command=self.next_slide)
        self.btnNext.place(x=900, y=10)

        #pc type classification list
        self.cellTypeDict = {'no-direction':0, 'x-mirror':1, 'remapping':2, 'remap-no-field':3, 'n/a':4}
        self.cellTypeList = list(self.cellTypeDict.keys())

        #cellType matrix
        self.cellType = np.zeros((self.numCells,))*np.nan
        self.cellTypeCorr = np.zeros((self.numCells,2))*np.nan
        self.cellTypeColor = ['#F28286','#F0CC6B','#9ECD53','#856BA8','#BDBDBD']

        #classify cells
        self.automatic_classification()

        #create plots 
        self.currSlide = 0
        self.create_plot()


        options = ['n/a','n/a']
        self.cellNumMenuValue = StringVar(window)
        self.cellNumMenuValue.set('Cell #') # default value
        self.cellNumMenu = OptionMenu(window, self.cellNumMenuValue, 
                                            *options, command=self.update_menuCellType)
        self.cellNumMenu.place(x=1000, y=10)


        self.cellTypeMenuValue = StringVar(window)
        self.cellTypeMenuValue.set('Cell type') # default value
        self.cellTypeMenu = OptionMenu(window, self.cellTypeMenuValue, 
                                            *self.cellTypeList, command=self.classify_cell)
        self.cellTypeMenu.place(x=1100, y=10)

        # #next cell slide buttom
        # self.btnNext=Button(window, text="Next", fg='black', command=self.next_slide)
        # self.btnNext.place(x=1000, y=10)
        self.next_slide()

    def update_menuCellType(self, value):
        cellType = self.cellType[value]
        self.cellTypeMenuValue.set(self.cellTypeList[cellType])

    def classify_cell(self, value):
        currCell = int(self.cellNumMenuValue.get())
        self.cellType[currCell] = self.cellTypeDict[value]
        print (f"Cell {currCell} type changed to: {value}")
        self.axHist.clear()
        self.plot_hist(self.axHist)
        plot_idx = int(currCell - (self.currSlide-1)*self.cellsPerSlide)
        # del self.fig.texts[4*plot_idx].set_color()
        self.fig.texts[3*plot_idx].set_color(self.cellTypeColor[int(self.cellTypeDict[value])])
        # self.set_cell_title(currCell, self.ax[plot_idx][0], 1.45)
        self.canvas.draw()
        np.save(os.path.join(self.saveDir, self.animalName+'_cellType.npy'), self.cellType)
        self.fig.savefig(os.path.join(self.saveDir, self.animalName+f'_plotCells_{self.currSlide}.png'), dpi = 400,bbox_inches="tight")

    def update_cellNumMenu(self):
        menu = self.cellNumMenu['menu']
        menu.delete(0,'end')
        #get global cell idx
        global_cell = (self.currSlide-1)*self.cellsPerSlide
        new_nums = [str(idx+global_cell) for idx in range(self.cellsPerSlide)]
        for string in new_nums:
            menu.add_command(label=string, 
                            command=lambda value=string: self.cellNumMenuValue.set(value))

    def automatic_classification(self):
        #check that mapAxis align
        mapAxis = self.PC['mapAxis'][0]
        mapAxisSize = mapAxis.shape[0]

        #classify each cell
        for cell in range(self.numCells):
                #get neuronal pdfs
                cell_pdf_r = self.PC['neu_pdf'][:,cell,0]

                cell_pdf_l = self.PC['neu_pdf'][:,cell,1]
                #check if cell is place-cell
                is_pc = self.PC['place_cells_dir'][cell]
                if np.sum(is_pc)==2: 
                    #compute correlations:
                    corr_og = np.corrcoef(cell_pdf_l.flatten(), cell_pdf_r.flatten())[0,1]
                    corr_xmirror = np.corrcoef(cell_pdf_l.flatten(), np.flipud(cell_pdf_r).flatten())[0,1]
                    self.cellTypeCorr[cell] = [corr_og,corr_xmirror]
                    firing_ratio = np.mean(cell_pdf_l)/np.mean(cell_pdf_r)
                    if (firing_ratio>=4)|(firing_ratio<=0.25): #big difference in firing rates: may be remapping
                        self.cellType[cell] = self.cellTypeDict['remapping']
                    elif np.all(self.cellTypeCorr[cell]<0.75): #low correlations: may be remapping
                        self.cellType[cell] = self.cellTypeDict['remapping']
                    else:
                        putative_type = np.argmax(self.cellTypeCorr[cell])
                        putative_type = ['no-direction', 'x-mirror', ][putative_type]
                        self.cellType[cell] = self.cellTypeDict[putative_type]
                elif np.sum(is_pc)==1: #only place cell in one of the directions
                    self.cellType[cell] = self.cellTypeDict['remap-no-field']
                else: #not a place cell
                    self.cellType[cell] = self.cellTypeDict['n/a']

        np.save(os.path.join(self.saveDir, self.animalName+'_cellType.npy'), self.cellType)
        np.save(os.path.join(self.saveDir, self.animalName+'_cellTypeCorr.npy'), self.cellTypeCorr)

    def next_slide(self):
        if self.currSlide==self.numSlides:
            self.currSlide = 1
        else:
            self.currSlide += 1

        # self.listboxCellType.selection_clear(0, 'end')
        self.titleCell.config(text = f"Slide {self.currSlide}/{self.numSlides}")
        self.plot_slide()

    def previous_slide(self):
        if self.currSlide==1:
            self.currSlide = self.numSlides
        else:
            self.currSlide -= 1

        # self.listboxCellType.selection_clear(0, 'end')
        self.titleCell.config(text = f"Slide {self.currSlide}/{self.numSlides}")
        self.plot_slide()

    def create_plot(self):

        self.fig = Figure(figsize=(20,8))
        gs = GridSpec(3*self.cellsPerColumn-1,5*self.cellsPerRow+3,figure = self.fig)
        self.ax = list()
        self.cbars = list()
        for idx in range(self.cellsPerSlide):
            self.ax.append(list([0,0]))
            self.cbars.append(0)
            rowIdx = idx//self.cellsPerRow
            colIdx = idx%self.cellsPerRow
            self.ax[idx][0] = self.fig.add_subplot(gs[3*rowIdx,5*colIdx:5*colIdx+3]) #scatter pre
            self.ax[idx][1] = self.fig.add_subplot(gs[3*rowIdx+1,5*colIdx:5*colIdx+3]) #heatmap pre


        self.axHist = self.fig.add_subplot(gs[:,-3:]) #cell classification hist
        self.fig.subplots_adjust(wspace=0.025, hspace=0.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(padx=50, pady=40, side=LEFT)
        self.fig.canvas.draw()

    def plot_slide(self):
        #get global cell idx
        global_cell = (self.currSlide-1)*self.cellsPerSlide

        for txt in range(len(self.fig.texts)):
            del self.fig.texts[0]

        for idx in range(self.cellsPerSlide):
            currCell = idx+global_cell
            if currCell>=self.numCells:
                continue
            neu_pdf = self.PC['neu_pdf'][:,currCell]

            maxVal = np.percentile(neu_pdf.flatten(),95)

            self.ax[idx][0].clear() #scatter
            self.plot_scatter(self.ax[idx][0], self.PC, currCell)
            self.ax[idx][1].clear() #heatmap
            # if not self.cbars[idx] == 0:
            #     self.cbars[idx].remove();
            self.plot_heatmap(self.ax[idx][1], self.PC,currCell, idx, maxVal)
            self.set_cell_title(currCell, self.ax[idx][0], 1.45)


        self.axHist.clear()
        self.plot_hist(self.axHist)
        self.canvas.draw()
        self.update_cellNumMenu()
        self.fig.savefig(os.path.join(self.saveDir, self.animalName+f'_plotCells_{self.currSlide}.png'), dpi = 400,bbox_inches="tight")

    def plot_hist(self, ax):
        cnts, values, bars = ax.hist(self.cellType, bins = [-0.5,0.5,1.5,2.5,3.5,4.5], 
                            rwidth=0.9, orientation="horizontal")
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(self.cellTypeList, rotation = -90, fontsize=12, va='center')
        ax.set_xlabel('Num Cells')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='x', alpha=0.75)
        ax.set_title('Cell Types Distribution')
        for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
            bar.set_facecolor(self.cellTypeColor[i % len(self.cellTypeColor)])

    def plot_scatter(self,ax,pc_struct, currCell):
        if currCell in pc_struct['place_cells_idx']:
            color_cell = [.2,.6,.2]
        else:
            color_cell = 'k'
        pos = pc_struct['pos_sig']
        direction = pc_struct['dir_sig']
        signal = pc_struct['neu_sig']

        temp_signal = signal[:,currCell]
        idxs = np.argsort(temp_signal)
        ordered_pos = pos[idxs,:]    
        ordered_signal = temp_signal[idxs]
        ordered_dir = direction[idxs]

        norm_cmap = matplotlib.colors.Normalize(vmin=0, vmax=1)
        point_color = list()
        for ii in range(temp_signal.shape[0]):
            #colormap possible values = viridis, jet, spectral
            if ordered_dir[ii]==np.unique(direction)[0]:
                point_color.append(np.array(plt.cm.viridis(norm_cmap(ordered_signal[ii]),bytes=True))/255)
            else:
                point_color.append(np.array(plt.cm.inferno(norm_cmap(ordered_signal[ii]),bytes=True))/255)
        ax.scatter(*ordered_pos.T, color = point_color, s= 8, alpha=1)
        ax.set_xlim([np.min(pos[:,0]), np.max(pos[:,0])])
        ax.set_ylim([np.min(pos[:,1]), np.max(pos[:,1])])
        title = list()
        [title.append(f"{mval:.2f} ") for midx, mval in enumerate(pc_struct['metric_val'][currCell])];
        ax.set_title(' | '.join(title), color = color_cell, fontsize = 8)

    def plot_heatmap(self, ax, pc_struct, currCell, idx, maxVal):
        # if not self.cbars[idx] == 0:
        #     self.cbars[idx].ax.clear();#remove();
        neu_pdf = pc_struct['neu_pdf']
        p = ax.matshow(neu_pdf[:,currCell].T, vmin = 0, vmax = maxVal, aspect = 'auto')
        ax.set_yticks([])
        ax.set_xticks([])
        # self.cbars[idx] = self.fig.colorbar(p, ax=ax,fraction=0.3, pad=0.08, location = 'bottom')

    def set_cell_title(self, currCell, ax, y):
        corr = self.cellTypeCorr[currCell]
        labelList = [f'Cell {currCell}:', f'og:{corr[0]:.2f}', f'xmirror:{corr[1]:.2f}']
        colorList = ['k']*3
        currCellType = self.cellType[currCell]
        colorList[0] =  self.cellTypeColor[int(currCellType)]
        if currCellType < 2:
            colorList[1+int(currCellType)] =  self.cellTypeColor[int(currCellType)]
        self.color_title_helper(labelList, colorList, ax, y=y)

    def color_title_helper(self,labels, colors, ax, y = 1.013, precision = 10**-3):
        "Creates a centered title with multiple colors. Don't change axes limits afterwards." 

        transform = ax.transAxes # use axes coords
        # initial params
        xT = 0# where the text ends in x-axis coords
        shift = 0 # where the text starts
        
        # for text objects
        text = dict()
        while (np.abs(shift - (2-xT)) > precision) and (shift <= xT) :         
            x_pos = shift 
            for label, col in zip(labels, colors):
                try:
                    text[label].remove()
                except KeyError:
                    pass
                if 'Cell' in label:
                    fontsize = 10
                else:
                    fontsize = 10
                text[label] = self.fig.text(x_pos, y, label, 
                            transform = transform, 
                            ha = 'left',
                            color = col,
                            fontsize = fontsize)
                
                x_pos = text[label].get_window_extent()\
                       .transformed(transform.inverted()).x1 + 0.15
                
            xT = x_pos # where all text ends
            shift += precision/1 # increase for next iteration
            if x_pos > 1: # guardrail 
                break


{'no-direction':0, 'x-mirror':1, 'remapping':2, 'remap-no-field':3, 'n/a':4}
#__________________________________________________________________________
#|                                                                        |#
#|                              PLACE CELLS                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'

params = {
    'sF': 20,
    'bin_width': 2.5,
    'std_pos': 0,
    'std_pdf': 5,
    'method': 'spatial_info',
    'num_shuffles': 1000,
    'min_shift': 10,
    'th_metric': 99,
    }

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)

    neuSignal = np.concatenate(animal['clean_traces'].values, axis=0)
    posSignal = np.concatenate(animal['pos'].values, axis=0)
    velSignal = np.concatenate(animal['vel'].values, axis=0)
    dirSignal = np.concatenate(animal['dir_mat'].values, axis=0)

    to_keep = np.logical_and(dirSignal[:,0]>0,dirSignal[:,0]<=2)
    posSignal = posSignal[to_keep,:] 
    velSignal = velSignal[to_keep] 
    neuSignal = neuSignal[to_keep,:] 
    dirSignal = dirSignal[to_keep,:] 

    mousePC = pc.get_place_cells(posSignal, neuSignal, vel_signal = velSignal, dim = 1,
                          direction_signal = dirSignal, mouse = mouse, save_dir = saveDir, **params)

    print('\tNum place cells:')
    num_cells = neuSignal.shape[1]
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*(mousePC['place_cells_dir'][:,1]==0))
    print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,1]*(mousePC['place_cells_dir'][:,0]==0))
    print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*mousePC['place_cells_dir'][:,1])
    print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    with open(os.path.join(saveDir,mouse+'_pc_dict.pkl'), 'wb') as f:
        pickle.dump(mousePC, f)
#__________________________________________________________________________
#|                                                                        |#
#|                         CLASSIFY PLACE CELLS                           |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/functional_cells/'
for mouse in mice_list:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_pc_dict.pkl'

    animal_pc = load_pickle(dataDir, file_name)
    savePath = os.path.join(saveDir,mouse)
    try:
        os.mkdir(savePath)
    except:
        pass

    window=Tk()
    myWin = windowClassifyPC(window, animal_pc, animalName = mouse, saveDir=savePath)
    window.title(f'Place Cell Classification for {mouse}')
    window.geometry("1500x800+10+20")
    window.mainloop()



#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT PLACE CELLS                             |#
#|________________________________________________________________________|#
import seaborn as sns

mice_list = ['GC2','GC3', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

sup_mice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']
data_dir = '/home/julio/Documents/SP_project/Fig2/functional_cells/'

percTypes_list = []
mouseName_list = []
mouseType_list = []
nameTypes_list = []
for mouse in mice_list:
    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(data_dir, mouse)

    cellType = np.load(os.path.join(file_path, file_name))

    num = [np.sum(cellType==0), np.sum(np.logical_and(cellType>1,cellType<4)), np.sum(cellType==4)]
    perc = [x/cellType.shape[0] for x in num]

    percTypes_list += perc
    mouseName_list += [mouse]*3
    if mouse in deep_mice:
        mouseType_list += ['deep']*3
    else:
        mouseType_list += ['sup']*3

    nameTypes_list += ['No-directional', 'Directional', 'N/A']


pd_cellTypes = pd.DataFrame(data={'mouseName': mouseName_list,
                             'percType': percTypes_list,
                             'mouseType': mouseType_list, 
                             'cellType': nameTypes_list})

palette= ["#cc9900ff", "#9900ffff"]



#_________________
fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType',
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)
#_________________
fig, ax = plt.subplots(1,1)
b = sns.barplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType',
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)




#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT PLACE CELLS                             |#
#|________________________________________________________________________|#
import seaborn as sns

mice_list = ['GC2','GC3', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

sup_mice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']
data_dir = '/home/julio/Documents/SP_project/Fig2/functional_cells/'

percTypes_list = []
mouseName_list = []
mouseType_list = []
nameTypes_list = []
for mouse in mice_list:
    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(data_dir, mouse)

    cellType = np.load(os.path.join(file_path, file_name))

    num = [np.sum(cellType==0), np.sum(cellType==1), np.sum(cellType==2), np.sum(cellType==3), np.sum(cellType==4)]
    perc = [x/cellType.shape[0] for x in num]

    percTypes_list += perc
    mouseName_list += [mouse]*5
    if mouse in deep_mice:
        mouseType_list += ['deep']*5
    else:
        mouseType_list += ['sup']*5

    nameTypes_list += ['No-directional', 'x-mirror', 'remap', 'remap-no-field', 'N/A']


pd_cellTypes = pd.DataFrame(data={'mouseName': mouseName_list,
                             'percType': percTypes_list,
                             'mouseType': mouseType_list, 
                             'cellType': nameTypes_list})

palette= ["#cc9900ff", "#9900ffff"]



#_________________
fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType',
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(data_dir,'field_types.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'field_types.png'), dpi = 400,bbox_inches="tight")



percTypes_list = []
mouseName_list = []
mouseType_list = []
nameTypes_list = []
for mouse in mice_list:
    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(data_dir, mouse)

    cellType = np.load(os.path.join(file_path, file_name))

    num = [np.sum(cellType==0), np.sum(cellType==1)+np.sum(cellType==2)+np.sum(cellType==3), np.sum(cellType==4)]
    perc = [x/cellType.shape[0] for x in num]

    percTypes_list += perc
    mouseName_list += [mouse]*3
    if mouse in deep_mice:
        mouseType_list += ['deep']*3
    else:
        mouseType_list += ['sup']*3

    nameTypes_list += ['no-directional', 'directional', 'N/A']


pd_cellTypes2 = pd.DataFrame(data={'mouseName': mouseName_list,
                             'percType': percTypes_list,
                             'mouseType': mouseType_list, 
                             'cellType': nameTypes_list})


#_________________
fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='cellType', y='percType', data=pd_cellTypes2, hue = 'mouseType',
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes2, hue = 'mouseType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(data_dir,'field_types_v2.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'field_types_v2.png'), dpi = 400,bbox_inches="tight")


fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='cellType', y='percType', data=pd_cellTypes2,
        linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes2, dodge = True,
            edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(data_dir,'field_types_v3.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'field_types_v3.png'), dpi = 400,bbox_inches="tight")




percTypes_list = []
mouseName_list = []
mouseType_list = []
nameTypes_list = []
for mouse in mice_list:
    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(data_dir, mouse)

    cellType = np.load(os.path.join(file_path, file_name))
    perc = np.sum(cellType<4)/cellType.shape[0]

    percTypes_list += [perc]
    mouseName_list += [mouse]
    if mouse in deep_mice:
        mouseType_list += ['deep']
    else:
        mouseType_list += ['sup']



pd_cellTypes3 = pd.DataFrame(data={'mouseName': mouseName_list,
                             'percType': percTypes_list,
                             'mouseType': mouseType_list})


#_________________

fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='mouseType', y='percType', data=pd_cellTypes3,
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='mouseType', y='percType', data=pd_cellTypes3, dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(data_dir,'field_types_v3.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'field_types_v3.png'), dpi = 400,bbox_inches="tight")

deepFields = pd_cellTypes3[pd_cellTypes3['mouseType']=='deep']['percType']
supFields = pd_cellTypes3[pd_cellTypes3['mouseType']=='sup']['percType']
deepShapiro = shapiro(deepFields)
supShapiro = shapiro(supFields)

if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('Fields:',stats.ks_2samp(deepFields, supFields))
else:
    print('Fields:', stats.ttest_ind(deepFields, supFields))
