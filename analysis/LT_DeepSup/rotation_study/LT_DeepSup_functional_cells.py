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
    def __init__(self,window, prePC, rotPC, animalName = None, saveDir=None):
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
        self.prePC = prePC
        self.rotPC = rotPC
        self.numCells = prePC['neu_sig'].shape[1]

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
        self.cellTypeDict = {'no-change':0, 'rotation':1, 'x-mirror':2, 'y-mirror':3, 'remapping':4, 'n/a':5}
        self.cellTypeList = list(self.cellTypeDict.keys())

        #cellType matrix
        self.cellType = np.zeros((self.numCells,))*np.nan
        self.cellTypeCorr = np.zeros((self.numCells,4))*np.nan
        self.cellTypeColor = ['#F28286','#F0CC6B','#9ECD53','#3D99D3','#856BA8','#BDBDBD']

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
        self.fig.texts[5*plot_idx].set_color(self.cellTypeColor[int(self.cellTypeDict[value])])
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
        preAxis = self.prePC['mapAxis'][0]
        rotAxis = self.rotPC['mapAxis'][0]
        preAxisSize = preAxis.shape[0]
        rotAxisSize = rotAxis.shape[0]

        diffSize = preAxisSize - rotAxisSize
        if diffSize<0: #pre shorter than rot
            leftAlign = abs(np.mean(rotAxis[:-diffSize] - preAxis))
            rightAlign = abs(np.mean(rotAxis[diffSize:] - preAxis))
            if leftAlign<=rightAlign:
                self.rotPC['neu_pdf'] = self.rotPC['neu_pdf'][:diffSize]
                self.rotPC['mapAxis'][0] = self.rotPC['mapAxis'][0][:diffSize]
            else:
                self.rotPC['neu_pdf'] = self.rotPC['neu_pdf'][-diffSize:]
                self.rotPC['mapAxis'][0] = self.rotPC['mapAxis'][0][-diffSize:]
        elif diffSize>0:
            leftAlign = abs(np.mean(preAxis[:-diffSize] - rotAxis))
            rightAlign = abs(np.mean(preAxis[diffSize:] - rotAxis))
            if leftAlign<=rightAlign:
                self.prePC['neu_pdf'] = self.prePC['neu_pdf'][:-diffSize]
                self.prePC['mapAxis'][0] = self.prePC['mapAxis'][0][:-diffSize]
            else:
                self.prePC['neu_pdf'] = self.prePC['neu_pdf'][diffSize:]
                self.prePC['mapAxis'][0] = self.prePC['mapAxis'][0][diffSize:]

        #classify each cell
        for cell in range(self.numCells):
                #get neuronal pdfs
                cell_pdf_p = self.prePC['neu_pdf'][:,cell]
                cell_pdf_r = self.rotPC['neu_pdf'][:,cell]
                #check if cell is place-cell
                is_pc_pre = cell in self.prePC['place_cells_idx']
                is_pc_rot = cell in self.rotPC['place_cells_idx']
                if is_pc_pre & is_pc_rot: 
                    #compute correlations:
                    corr_og = np.corrcoef(cell_pdf_p.flatten(), cell_pdf_r.flatten())[0,1]
                    corr_rot = np.corrcoef(cell_pdf_p.flatten(), np.flipud(np.fliplr(cell_pdf_r)).flatten())[0,1]
                    corr_xmirror = np.corrcoef(cell_pdf_p.flatten(), np.flipud(cell_pdf_r).flatten())[0,1]
                    corr_ymirror = np.corrcoef(cell_pdf_p.flatten(), np.fliplr(cell_pdf_r).flatten())[0,1]
                    self.cellTypeCorr[cell] = [corr_og,corr_rot, corr_xmirror, corr_ymirror]

                    firing_ratio = np.mean(cell_pdf_p)/np.mean(cell_pdf_r)
                    if (firing_ratio>=4)|(firing_ratio<=0.25): #big difference in firing rates: may be remapping
                        self.cellType[cell] = self.cellTypeDict['remapping']
                    elif np.all(self.cellTypeCorr[cell]<0.75): #low correlations: may be remapping
                        self.cellType[cell] = self.cellTypeDict['remapping']
                    else:
                        putative_type = np.argmax(self.cellTypeCorr[cell])
                        putative_type = ['no-change', 'rotation','x-mirror', 'y-mirror'][putative_type]
                        self.cellType[cell] = self.cellTypeDict[putative_type]

                elif is_pc_pre^is_pc_rot: #only place cell in one of the sessions
                    self.cellType[cell] = self.cellTypeDict['remapping']
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
        gs = GridSpec(3*self.cellsPerColumn-1,9*self.cellsPerRow+4,figure = self.fig)
        self.ax = list()
        self.cbars = list()
        for idx in range(self.cellsPerSlide):
            self.ax.append(list([0,0,0,0]))
            self.cbars.append(list([0,0]))
            rowIdx = idx//self.cellsPerRow
            colIdx = idx%self.cellsPerRow
            self.ax[idx][0] = self.fig.add_subplot(gs[3*rowIdx,9*colIdx:9*colIdx+3]) #scatter pre
            self.ax[idx][1] = self.fig.add_subplot(gs[3*rowIdx,9*colIdx+4:9*colIdx+7]) #scatter rot
            self.ax[idx][2] = self.fig.add_subplot(gs[3*rowIdx+1,9*colIdx:9*colIdx+3]) #heatmap pre
            self.ax[idx][3] = self.fig.add_subplot(gs[3*rowIdx+1,9*colIdx+4:9*colIdx+7]) #heatmap rot


        self.axHist = self.fig.add_subplot(gs[:,-4:]) #cell classification hist
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
            neu_pdf_p = self.prePC['neu_pdf'][:,currCell]
            neu_pdf_r = self.rotPC['neu_pdf'][:,currCell]
            maxVal = np.percentile(np.concatenate((neu_pdf_p.flatten(),neu_pdf_r.flatten())),95)

            self.ax[idx][0].clear() #scatter pre
            self.plot_scatter(self.ax[idx][0], self.prePC, currCell)

            self.ax[idx][1].clear() #scatter rot
            self.plot_scatter(self.ax[idx][1], self.rotPC, currCell)

            self.ax[idx][2].clear() #heatmap pre
            self.plot_heatmap(self.ax[idx][2], self.prePC,0, currCell, idx, maxVal)

            self.ax[idx][3].clear() #heatmap pre
            self.plot_heatmap(self.ax[idx][3], self.rotPC,1, currCell, idx, maxVal)

            self.set_cell_title(currCell, self.ax[idx][0], 1.45)
            # cellTypeIdx = int(self.cellType[currCell])
            # self.valueInsideMenus[idx].set(self.cellTypeList[cellTypeIdx])
            # self.menuCellType[idx].config(fg=self.cellTypeColor[cellTypeIdx])

        self.axHist.clear()
        self.plot_hist(self.axHist)
        self.canvas.draw()
        self.update_cellNumMenu()
        self.fig.savefig(os.path.join(self.saveDir, self.animalName+f'_plotCells_{self.currSlide}.png'), dpi = 400,bbox_inches="tight")

    def plot_hist(self, ax):
        cnts, values, bars = ax.hist(self.cellType, bins = [-0.5, 0.5, 1.5,2.5,3.5,4.5,5.5,5.5], 
                            rwidth=0.9, orientation="horizontal")
        ax.set_yticks(np.arange(6))
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

    def plot_heatmap(self, ax, pc_struct,num, currCell, idx, maxVal):
        try:
            print(cbars[idx][num])
            if not self.cbars[idx][num] == 0:
                self.cbars[idx][num].remove()
        except:
            a = 0
        neu_pdf = pc_struct['neu_pdf']
        p = ax.matshow(neu_pdf[:,currCell].T, vmin = 0, vmax = maxVal, aspect = 'auto')
        ax.set_yticks([])
        ax.set_xticks([])
        self.cbars[idx][num] = self.fig.colorbar(p, ax=ax,fraction=0.3, pad=0.08, location = 'bottom')

    def set_cell_title(self, currCell, ax, y):
        corr = self.cellTypeCorr[currCell]
        labelList = [f'Cell {currCell}:', f'og:{corr[0]:.2f}', f'rot:{corr[1]:.2f}', f'xmirror:{corr[2]:.2f}', f'ymirror:{corr[3]:.2f}']
        colorList = ['k']*5
        currCellType = self.cellType[currCell]
        colorList[0] =  self.cellTypeColor[int(currCellType)]
        if currCellType < 4:
            colorList[1+int(currCellType)] =  self.cellTypeColor[int(currCellType)]
        self.color_title_helper(labelList, colorList, ax, y=y)

    def color_title_helper(self,labels, colors, ax, y = 1.013, precision = 10**-3):
        "Creates a centered title with multiple colors. Don't change axes limits afterwards." 

        transform = ax.transAxes # use axes coords
        # initial params
        xT = 0# where the text ends in x-axis coords
        shift = -0.1 # where the text starts
        
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
                    fontsize = 8
                text[label] = self.fig.text(x_pos, y, label, 
                            transform = transform, 
                            ha = 'left',
                            color = col,
                            fontsize = fontsize)
                
                x_pos = text[label].get_window_extent()\
                       .transformed(transform.inverted()).x1 + 0.15
                
            xT = x_pos # where all text ends
            shift += precision/2 # increase for next iteration
            if x_pos > 2: # guardrail 
                break

#__________________________________________________________________________
#|                                                                        |#
#|                         CLASSIFY PLACE CELLS                           |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
for mouse in mice_list:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_pc_dict.pkl'

    animal_pc = load_pickle(dataDir, file_name)
    fnames = list(animal_pc.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    savePath = os.path.join(saveDir,mouse)
    try:
        os.mkdir(savePath)
    except:
        pass

    window=Tk()
    myWin = windowClassifyPC(window, animal_pc[fnamePre], animal_pc[fnameRot], animalName = mouse, saveDir=savePath)
    window.title(f'Place Cell Classification for {mouse}')
    window.geometry("1500x800+10+20")
    window.mainloop()


#######################################################################

deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
sup_mice = ['ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
percTypes_list = []
mouseName_list = []
mouseType_list = []
nameTypes_list = []
for mouse in mice_list:
    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(data_dir, mouse)

    cellType = np.load(os.path.join(file_path, file_name))

    num = [np.sum(cellType==0), np.sum(np.logical_and(cellType>0,cellType<4)), np.sum(cellType==4), np.sum(cellType==5)]
    perc = [x/cellType.shape[0] for x in num]

    percTypes_list += perc
    mouseName_list += [mouse]*4
    if mouse in deep_mice:
        mouseType_list += ['deep']*4
    else:
        mouseType_list += ['sup']*4

    nameTypes_list += ['Global', 'Local', 'Remap', 'N/A']


pd_cellTypes = pd.DataFrame(data={'mouseName': mouseName_list,
                             'percType': percTypes_list,
                             'mouseType': mouseType_list, 
                             'cellType': nameTypes_list})

palette= ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType',
        palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cellType', y='percType', data=pd_cellTypes, hue = 'mouseType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)



from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('percType ~ C(cellType) + C(mouseType) + C(cellType):C(mouseType)', data=pd_cellTypes).fit()
sm.stats.anova_lm(model, typ=2)




fig, ax = plt.subplots(1,1)
b = sns.boxplot(x='mouseType', y='percType', data=pd_cellTypes, hue = 'cellType',
        palette = palette, linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='mouseType', y='percType', data=pd_cellTypes, hue = 'cellType', dodge = True,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax)

#__________________________________________________________________________
#|                                                                        |#
#|                         CLASSIFY PLACE CELLS                           |#
#|________________________________________________________________________|#

mice_list = ['ChCharly1', 'ChCharly2']
data_dir = '/media/julio/DATOS/spatial_navigation/paper/Fig4/place_cells'

for mouse in mice_list:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_pc_dict.pkl'
    file_path = os.path.join(data_dir, mouse)

    animal_pc = load_pickle(file_path, file_name)
    fnames = list(animal_pc.keys())

    window=Tk()
    myWin = windowClassifyPC(window, animal_pc[fnames[0]], animal_pc[fnames[1]], animalName = mouse, saveDir=file_path)
    window.title(f'Place Cell Classification for {mouse}')
    window.geometry("1500x800+10+20")
    window.mainloop()
