import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neural_manifold import place_cells as pc


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))


#__________________________________________________________________________
#|                                                                        |#
#|                              PLACE CELLS                               |#
#|________________________________________________________________________|#

mouse = 'CalbCharly11_concat'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

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


print(f"Working on mouse: {mouse}")
for case in ['veh', 'CNO']:
    print(f"\tCondition: {case}")
    file_name =  f"{mouse}_{case}_df_dict.pkl"
    case_dir = os.path.join(base_dir, 'processed_data', mouse+'_'+case)
    save_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    mouse_place_cells = {}
    animal = load_pickle(case_dir,file_name)
    fnames = list(animal.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]

    for fname in [fname_pre, fname_rot]:

        animal_session= copy.deepcopy(animal[fname])
        pos = get_signal(animal_session, 'position')        
        traces = get_signal(animal_session, 'clean_traces')        
        vel = get_signal(animal_session, 'speed')
        mov_dir = get_signal(animal_session, 'mov_direction')


        to_keep = mov_dir!=0
        pos = pos[to_keep,:] 
        vel = vel[to_keep] 
        traces = traces[to_keep,:] 
        mov_dir = mov_dir[to_keep].reshape(-1,1)

        mouse_place_cells[fname] = pc.get_place_cells(pos, traces, vel_signal = vel, dim = 1,
                              direction_signal = mov_dir, mouse = fname, save_dir = save_dir, **params)

        print(f'\t{fname} num place cells:')
        num_cells = traces.shape[1]
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,0]*(mouse_place_cells[fname]['place_cells_dir'][:,1]==0))
        print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,1]*(mouse_place_cells[fname]['place_cells_dir'][:,0]==0))
        print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,0]*mouse_place_cells[fname]['place_cells_dir'][:,1])
        print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')

        with open(os.path.join(save_dir,mouse+'_'+case+'_pc_dict.pkl'), 'wb') as f:
            pickle.dump(mouse_place_cells, f)


#__________________________________________________________________________
#|                                                                        |#
#|                          PLACE CELLS REGISTERED                        |#
#|________________________________________________________________________|#
mice_list = ['CalbCharly2', 'DD2', 'CalbV23']
mouse = 'CalbV23'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

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


print(f"Working on mouse: {mouse}")


for case in ['veh', 'CNO']:
    print(f"\tCondition: {case}")
    file_name =  f"{mouse}_{case}_df_dict.pkl"
    case_dir = os.path.join(base_dir, 'processed_data', mouse+'_'+case)
    save_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    mouse_place_cells = {}
    animal = load_pickle(case_dir,file_name)
    fnames = list(animal.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]

    cellmap = pd.read_csv(os.path.join(case_dir, f"{mouse}_veh_CNO_cellmap.csv"))
    registered_cells = cellmap[case+'_idx'].to_list()
    for fname in [fname_pre, fname_rot]:

        animal_session= copy.deepcopy(animal[fname])
        pos = get_signal(animal_session, 'position')        
        traces = get_signal(animal_session, 'clean_traces')[:, registered_cells]        
        vel = get_signal(animal_session, 'speed')
        mov_dir = get_signal(animal_session, 'mov_direction')


        to_keep = mov_dir!=0
        pos = pos[to_keep,:] 
        vel = vel[to_keep] 
        traces = traces[to_keep,:] 
        mov_dir = mov_dir[to_keep].reshape(-1,1)


        #keep only registered cells
        mouse_place_cells[fname] = pc.get_place_cells(pos, traces, vel_signal = vel, dim = 1,
                              direction_signal = mov_dir, mouse = fname, save_dir = save_dir, **params)

        print(f'\t{fname} num place cells:')
        num_cells = traces.shape[1]
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,0]*(mouse_place_cells[fname]['place_cells_dir'][:,1]==0))
        print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,1]*(mouse_place_cells[fname]['place_cells_dir'][:,0]==0))
        print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mouse_place_cells[fname]['place_cells_dir'][:,0]*mouse_place_cells[fname]['place_cells_dir'][:,1])
        print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')

        with open(os.path.join(save_dir,mouse+'_'+case+'_pc_dict.pkl'), 'wb') as f:
            pickle.dump(mouse_place_cells, f)


#__________________________________________________________________________
#|                                                                        |#
#|                         FUNCTIONAL CELLS                               |#
#|________________________________________________________________________|#
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
                #compute correlations:

                corr_og = np.corrcoef(cell_pdf_p.flatten(), cell_pdf_r.flatten())[0,1]
                corr_rot = np.corrcoef(cell_pdf_p.flatten(), np.flipud(np.fliplr(cell_pdf_r)).flatten())[0,1]
                corr_xmirror = np.corrcoef(cell_pdf_p.flatten(), np.flipud(cell_pdf_r).flatten())[0,1]
                corr_ymirror = np.corrcoef(cell_pdf_p.flatten(), np.fliplr(cell_pdf_r).flatten())[0,1]
                self.cellTypeCorr[cell] = [corr_og,corr_rot, corr_xmirror, corr_ymirror]
                if is_pc_pre & is_pc_rot: 
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
        # self.cbars[idx][num] = self.fig.colorbar(p, ax=ax,fraction=0.3, pad=0.08, location = 'bottom')

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
mice_list = ['CalbCharly11_concat', 'CalbCharly2', 'DD2', 'CalbV23']

mouse = 'CalbV23'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

print(f"Working on mouse: {mouse}")
for case in ['veh', 'CNO']:

    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_'+case+'_pc_dict.pkl'
    data_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
    animal_pc = load_pickle(data_dir, file_name)

    fnames = list(animal_pc.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]

    save_dir = os.path.join(base_dir, 'functional_cells', mouse+'_'+case)
    try:
        os.mkdir(save_dir)
    except:
        pass

    window=Tk()
    myWin = windowClassifyPC(window, animal_pc[fname_pre], animal_pc[fname_rot], animalName = f"{mouse}_{case}", saveDir=save_dir)
    window.title(f'Place Cell Classification for {mouse}')
    window.geometry("1500x800+10+20")
    window.mainloop()

#__________________________________________________________________________
#|                                                                        |#
#|                  CREATE PD OF CELLS VEH + CNO CLASS                    |#
#|________________________________________________________________________|#
import seaborn as sns

mice_list = ['CalbCharly2', 'CalbCharly11_concat', 'DD2', 'CalbV23']
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

veh_cell_type_list = list()
cno_cell_type_list = list()

veh_activity_pre_list = list()
veh_activity_rot_list = list()
cno_activity_pre_list = list()
cno_activity_rot_list = list()


veh_metric_pre_list = list()
veh_metric_rot_list = list()
cno_metric_pre_list = list()
cno_metric_rot_list = list()

animal_list = list()
for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    for case in ['veh', 'CNO']:
        file_name =  mouse+'_'+case+'_pc_dict.pkl'
        data_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
        animal_pc = load_pickle(data_dir, file_name)

        fnames = list(animal_pc.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]

        file_name =  mouse+'_'+case+'_cellType.npy'
        data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_'+case)
        animal_cell_class = np.load(os.path.join(data_dir, file_name))

        for cell in range(animal_pc[fname_pre]['metric_val'].shape[0]):
            activity_pre = 20*np.sum(animal_pc[fname_pre]['total_firing_neurons'][cell,:])/animal_pc[fname_pre]['vel_sig'].shape[0]
            activity_rot = 20*np.sum(animal_pc[fname_rot]['total_firing_neurons'][cell,:])/animal_pc[fname_rot]['vel_sig'].shape[0]

            metric_pre = np.sum(animal_pc[fname_pre]['metric_val'][cell,:])
            metric_rot = np.sum(animal_pc[fname_rot]['metric_val'][cell,:])

            cell_type = animal_cell_class[cell]
            cell_type = np.where([cell_type==0, np.logical_and(cell_type>0,cell_type<4), cell_type==4, cell_type==5])[0][0]
            cell_type = ['global', 'local', 'remapping', 'other'][cell_type]
            if 'veh' in case:
                animal_list.append(mouse)
                veh_activity_pre_list.append(activity_pre)
                veh_activity_rot_list.append(activity_rot)
                veh_metric_pre_list.append(metric_pre)
                veh_metric_rot_list.append(metric_rot)
                veh_cell_type_list.append(cell_type)
            elif 'CNO' in case:
                cno_activity_pre_list.append(activity_pre)
                cno_activity_rot_list.append(activity_rot)
                cno_metric_pre_list.append(metric_pre)
                cno_metric_rot_list.append(metric_rot)
                cno_cell_type_list.append(cell_type)


pd_cell_type = pd.DataFrame(data={
                             'mouse': animal_list,
                             'veh_cell_type': veh_cell_type_list,
                             'veh_activity_pre': veh_activity_pre_list,
                             'veh_activity_rot': veh_activity_rot_list,
                             'veh_metric_pre': veh_metric_pre_list,
                             'veh_metric_rot': veh_metric_rot_list,

                             'cno_cell_type': cno_cell_type_list,
                             'cno_activity_pre': cno_activity_pre_list,
                             'cno_activity_rot': cno_activity_rot_list,
                             'cno_metric_pre': cno_metric_pre_list,
                             'cno_metric_rot': cno_metric_rot_list

                             })


#percentage of cells by type and condition
veh_cell = pd_cell_type.groupby(['veh_cell_type', 'mouse'], as_index=False).count().loc[:,['veh_cell_type', 'mouse', 'veh_activity_pre']]
veh_cell.rename(columns={"veh_activity_pre":"perc_cells"}, inplace = True)
veh_cell.rename(columns={"veh_cell_type":"cell_type"}, inplace = True)
num_cells_by_animal = veh_cell.groupby('mouse')['perc_cells'].sum()
for entry in veh_cell.index:
    veh_cell.loc[entry,'perc_cells'] = 100*veh_cell.loc[entry]['perc_cells'].copy()/num_cells_by_animal.loc[veh_cell.loc[entry]['mouse']].copy()
veh_cell['condition'] = 'veh'

cno_cell = pd_cell_type.groupby(['cno_cell_type', 'mouse'],as_index=False).count().loc[:,['cno_cell_type', 'mouse', 'cno_activity_pre']]
cno_cell.rename(columns={"cno_activity_pre":"perc_cells"}, inplace = True)
cno_cell.rename(columns={"cno_cell_type":"cell_type"}, inplace = True)

num_cells_by_animal = cno_cell.groupby('mouse')['perc_cells'].sum()
for entry in cno_cell.index:
    cno_cell.loc[entry,'perc_cells'] = 100*cno_cell.loc[entry]['perc_cells'].copy()/num_cells_by_animal.loc[cno_cell.loc[entry]['mouse']].copy()
cno_cell['condition'] = 'cno'

plot_pd = pd.concat([veh_cell, cno_cell])
plot_pd.reset_index(inplace=True)
plt.figure()
ax = plt.subplot(1,1,1)
sns.barplot(data=plot_pd, x='cell_type', y='perc_cells', hue='condition', ax=ax)
sns.swarmplot(data=plot_pd, x='cell_type', y='perc_cells', hue='condition', ax=ax, palette="dark:gray")


plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_barplot.svg"), dpi=400, bbox_inches="tight")    
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_barplot.png"), dpi=400, bbox_inches="tight")



from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

#perform two-way ANOVA
model = ols('perc_cells ~ C(cell_type) + C(condition) + C(cell_type):C(condition)', data=plot_pd).fit()
print(sm.stats.anova_lm(model, typ=2))
track = []
for cell_type in plot_pd['cell_type'].unique():
    for cell_type2 in plot_pd['cell_type'].unique():
        for condition in plot_pd['condition'].unique():
            for condition2 in plot_pd['condition'].unique():
                if cell_type==cell_type2 and condition==condition2: continue;
                if (cell_type, condition, cell_type2, condition2) in track: continue;
                if (cell_type2, condition2, cell_type, condition)in track: continue;
                track.append((cell_type, condition, cell_type2, condition2))
                a = plot_pd[np.logical_and(plot_pd['cell_type']==cell_type,plot_pd['condition']==condition)]['perc_cells'].tolist()
                b = plot_pd[np.logical_and(plot_pd['cell_type']==cell_type2,plot_pd['condition']==condition2)]['perc_cells'].tolist()
                print(f'({cell_type},{condition}) vs ({cell_type2},{condition2}):', stats.ttest_rel(a, b))






#transition mat
transition_mat = np.zeros((4,4))*np.nan
for row, veh_type in enumerate(['global', 'local', 'remapping', 'other']):
    veh_num_cells = np.sum(pd_cell_type["veh_cell_type"]==veh_type)
    veh_pd = pd_cell_type[pd_cell_type["veh_cell_type"]==veh_type]
    for col, cno_type in enumerate(['global', 'local', 'remapping', 'other']):
        transition_num = np.sum(veh_pd["cno_cell_type"]==cno_type)
        transition_mat[row, col] = transition_num/veh_num_cells

fig = plt.figure()
ax = plt.subplot(1,1,1)
b = ax.matshow(transition_mat, vmin = 0, vmax = 0.75, cmap = plt.cm.viridis)
ax.xaxis.set_ticks_position('bottom')
cbar = fig.colorbar(b,ax=ax,anchor=(0, 0.2), shrink=1, ticks=[0,0.25,0.5, 0.75])
for (i, j), z in np.ndenumerate(transition_mat):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color = 'white')
ax.set_xticklabels(['']+['global', 'local', 'remapping', 'other'])
ax.set_yticklabels(['']+['global', 'local', 'remapping', 'other'])
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_transtion_mat.svg"), dpi=400, bbox_inches="tight")    
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_transtion_mat.png"), dpi=400, bbox_inches="tight")

#transition mat
transition_mat = np.zeros((3,3))*np.nan
for row, veh_type in enumerate(['global', 'local', 'remapping']):
    veh_num_cells = np.sum(pd_cell_type["veh_cell_type"]==veh_type)
    veh_pd = pd_cell_type[pd_cell_type["veh_cell_type"]==veh_type]

    cno_other = veh_pd[veh_pd["cno_cell_type"]=='other'].shape[0]
    veh_num_cells -= cno_other
    for col, cno_type in enumerate(['global', 'local', 'remapping']):
        transition_num = np.sum(veh_pd["cno_cell_type"]==cno_type)
        transition_mat[row, col] = transition_num/veh_num_cells

fig = plt.figure()
ax = plt.subplot(1,1,1)
b = ax.matshow(transition_mat, vmin = 0, vmax = 0.85, cmap = plt.cm.viridis)
ax.xaxis.set_ticks_position('bottom')
cbar = fig.colorbar(b,ax=ax,anchor=(0, 0.2), shrink=1, ticks=[0,0.25,0.5, 0.75, 1])
for (i, j), z in np.ndenumerate(transition_mat):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color = 'white')
ax.set_xticklabels(['']+['global', 'local', 'remapping'])
ax.set_yticklabels(['']+['global', 'local', 'remapping'])
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_transtion_mat_v2.svg"), dpi=400, bbox_inches="tight")    
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_transtion_mat_v2.png"), dpi=400, bbox_inches="tight")

#examples 
file_name =  mouse+'_veh_pc_dict.pkl'
data_dir = os.path.join(base_dir, 'place_cells', mouse+'_veh')
veh_pc = load_pickle(data_dir, file_name)
fnames = list(veh_pc.keys())
fname_pre = [fname for fname in fnames if 'lt' in fname][0]
fname_rot = [fname for fname in fnames if 'rot' in fname][0]
veh_pc_pre = veh_pc[fname_pre]['neu_pdf']
veh_pc_rot = veh_pc[fname_rot]['neu_pdf']

file_name =  mouse+'_CNO_pc_dict.pkl'
data_dir = os.path.join(base_dir, 'place_cells', mouse+'_CNO')
cno_pc = load_pickle(data_dir, file_name)
fnames = list(cno_pc.keys())
fname_pre = [fname for fname in fnames if 'lt' in fname][0]
fname_rot = [fname for fname in fnames if 'rot' in fname][0]
cno_pc_pre = cno_pc[fname_pre]['neu_pdf']
cno_pc_rot = cno_pc[fname_rot]['neu_pdf']


examples = {
    'global-2-remap': 121,
    'local-2-local': 161, #166,206,207,427,554
    'remap-2-remap': 544
}

max_vals = {
    'global-2-remap': [0.020, 0.28],
    'local-2-local': [0.09, 0.15],
    'remap-2-remap': [0.25,0.25]
}


fig, ax = plt.subplots(2,6, figsize=(14,3))
for col, (name, cell_idx) in enumerate(examples.items()):
    print(col, name, cell_idx)
    # max_val = np.percentile(np.concatenate((veh_pc_pre[:,cell_idx,:].flatten(),veh_pc_rot[:,cell_idx,:].flatten())),95)
    p1 = ax[0,2*col].matshow(veh_pc_pre[:,cell_idx,:].T, vmin = 0, vmax = max_vals[name][0], aspect = 'auto')
    p2 = ax[0,2*col+1].matshow(veh_pc_rot[:,cell_idx,:].T, vmin = 0, vmax = max_vals[name][0], aspect = 'auto')
    ax[0,2*col].set_yticks([])
    ax[0,2*col].set_xticks([])
    ax[0,2*col+1].set_yticks([])
    ax[0,2*col+1].set_xticks([])

    # max_val = np.percentile(np.concatenate((cno_pc_pre[:,cell_idx,:].flatten(),cno_pc_rot[:,cell_idx,:].flatten())),95)
    p3 = ax[1,2*col].matshow(cno_pc_pre[:,cell_idx,:].T, vmin = 0, vmax = max_vals[name][1], aspect = 'auto')
    p4 = ax[1,2*col+1].matshow(cno_pc_rot[:,cell_idx,:].T, vmin = 0, vmax = max_vals[name][1], aspect = 'auto')

    ax[1,2*col].set_yticks([])
    ax[1,2*col].set_xticks([])
    ax[1,2*col+1].set_yticks([])
    ax[1,2*col+1].set_xticks([])

plt.tight_layout()

plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_examples.svg"), dpi=400, bbox_inches="tight")    
plt.savefig(os.path.join(base_dir, 'functional_cells', f"functional_cells_examples.png"), dpi=400, bbox_inches="tight")



#clean some cell clses
veh_type = 'remapping'
cno_type = 'remapping'
veh_pd = pd_cell_type[pd_cell_type["veh_cell_type"] == veh_type]
veh_pd[veh_pd["cno_cell_type"] == cno_type]
#transition mat
transition_mat = np.zeros((4,4))*np.nan
for row, veh_type in enumerate(['global', 'local', 'remapping', 'other']):
    veh_num_cells = np.sum(pd_cell_type["veh_cell_type"]==veh_type)
    veh_pd = pd_cell_type[pd_cell_type["veh_cell_type"]==veh_type]
    for col, cno_type in enumerate(['global', 'local', 'remapping', 'other']):
        transition_num = np.sum(veh_pd["cno_cell_type"]==cno_type)
        transition_mat[row, col] = transition_num
fig = plt.figure()
ax = plt.subplot(1,1,1)
b = ax.matshow(transition_mat, vmin = 0, vmax = 150, cmap = plt.cm.viridis)
ax.xaxis.set_ticks_position('bottom')
cbar = fig.colorbar(b,ax=ax,anchor=(0, 0.2), shrink=1, ticks=[0,0.25,0.5, 0.75])
for (i, j), z in np.ndenumerate(transition_mat):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color = 'white')
ax.set_xticklabels(['']+['global', 'local', 'remapping', 'other'])
ax.set_yticklabels(['']+['global', 'local', 'remapping', 'other'])



#__________________________________________________________________________
#|                                                                        |#
#|                    FUNCTIONAL CELLS VS RING DISTANCE                   |#
#|________________________________________________________________________|#

rotation_pd['remap_perc'] = np.nan
# rotation_pd['veh_perc'] = np.nan
# rotation_pd['cno_perc'] = np.nan
rotation_pd['perc_change'] = np.nan
rotation_pd['dist_change'] = np.nan
for idx in range(rotation_pd.shape[0]):
    mouse = rotation_pd.loc[idx, 'mouse']
    condition = rotation_pd.loc[idx, 'case']

    mouse_pd = plot_pd[plot_pd['mouse']==mouse]
    remapping_pd = mouse_pd[mouse_pd['cell_type']=='remapping']
    perc = remapping_pd[remapping_pd['condition']==condition.lower()]['perc_cells'].iloc[0]
    rotation_pd['remap_perc'][idx] = perc
    veh = remapping_pd[remapping_pd['condition']=='veh']['perc_cells'].iloc[0]
    cno = remapping_pd[remapping_pd['condition']=='cno']['perc_cells'].iloc[0]
    # rotation_pd['veh_perc'][idx] = veh
    # rotation_pd['cno_perc'][idx] = cno
    rotation_pd['perc_change'][idx] = (cno - veh)/veh

    mouse_pd = rotation_pd[rotation_pd['mouse']==mouse]
    veh_dist = mouse_pd[mouse_pd['case']=='veh']['remap_dist'].iloc[0]
    cno_dist = mouse_pd[mouse_pd['case']=='CNO']['remap_dist'].iloc[0]
    rotation_pd['dist_change'][idx] = (cno_dist - veh_dist)/veh_dist


from scipy import stats


fig, ax = plt.subplots(1, 2, figsize=(15,6))
b = sns.scatterplot(x='remap_perc', y='remap_dist', data=rotation_pd,
        hue='case', style = 'mouse', edgecolor = 'gray', ax = ax[0])

temp_pd = rotation_pd[rotation_pd['case']=='CNO']
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_pd["perc_change"].to_list(),temp_pd["dist_change"].to_list())
b = sns.scatterplot(x='perc_change', y='dist_change', data=temp_pd,
        style = 'mouse', edgecolor = 'gray', ax = ax[1])
ax[1].plot([0.35,0.7], [intercept, slope+intercept], 'k--')
ax[1].set_title(stats.pearsonr(temp_pd["perc_change"].to_list(),temp_pd["dist_change"].to_list()))
#ROTATION



#__________________________________________________________________________
#|                                                                        |#
#|                        CORRECT MISCLASSIFICATIONS                      |#
#|________________________________________________________________________|#

import seaborn as sns

mouse = 'CalbCharly11_concat'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

#examples 
file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
veh_pc = np.load(os.path.join(data_dir, file_name))

file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
cno_pc = np.load(os.path.join(data_dir, file_name))


cno_pairs = [(73,2), (138,2), (148,2), (149,3), (176,1), (186,0), (205,2), (214,3), (226,0), (240,1), (260,0), (312,3), (331,3), (333,3), (382,0), (479,5)]
veh_pairs = [(134,1), (163,1), (186,0), (228,5), (253,3), (333,3), (349,5), (296,1), (438,1), (446,1), (467,1)]

for (cell_idx, cell_type) in veh_pairs:
    veh_pc[cell_idx] = cell_type

for (cell_idx, cell_type) in cno_pairs:
    cno_pc[cell_idx] = cell_type



file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
np.save(os.path.join(data_dir,file_name), veh_pc)



file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
np.save(os.path.join(data_dir,file_name), cno_pc)


print(f"Working on mouse: {mouse}")
for case in ['veh', 'CNO']:
    file_name =  mouse+'_'+case+'_pc_dict.pkl'
    data_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
    animal_pc = load_pickle(data_dir, file_name)

    fnames = list(animal_pc.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]

    file_name =  mouse+'_'+case+'_cellType.npy'
    data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_'+case)
    animal_cell_class = np.load(os.path.join(data_dir, file_name))

    for cell in range(animal_pc[fname_pre]['metric_val'].shape[0]):
        activity_pre = 20*np.sum(animal_pc[fname_pre]['total_firing_neurons'][cell,:])/animal_pc[fname_pre]['vel_sig'].shape[0]
        activity_rot = 20*np.sum(animal_pc[fname_rot]['total_firing_neurons'][cell,:])/animal_pc[fname_rot]['vel_sig'].shape[0]

        metric_pre = np.sum(animal_pc[fname_pre]['metric_val'][cell,:])
        metric_rot = np.sum(animal_pc[fname_rot]['metric_val'][cell,:])

        cell_type = animal_cell_class[cell]
        cell_type = np.where([cell_type==0, np.logical_and(cell_type>0,cell_type<4), cell_type==4, cell_type==5])[0][0]
        cell_type = ['global', 'local', 'remapping', 'other'][cell_type]
        if 'veh' in case:
            veh_activity_pre_list.append(activity_pre)
            veh_activity_rot_list.append(activity_rot)
            veh_metric_pre_list.append(metric_pre)
            veh_metric_rot_list.append(metric_rot)
            veh_cell_type_list.append(cell_type)
        elif 'CNO' in case:
            cno_activity_pre_list.append(activity_pre)
            cno_activity_rot_list.append(activity_rot)
            cno_metric_pre_list.append(metric_pre)
            cno_metric_rot_list.append(metric_rot)
            cno_cell_type_list.append(cell_type)




mouse = 'CalbCharly2'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

#examples 
file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
veh_pc = np.load(os.path.join(data_dir, file_name))

file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
cno_pc = np.load(os.path.join(data_dir, file_name))

cno_pairs = [(10,5)]
veh_pairs = [(10,5), (15,5), (25,5), (57,5), (65,5), (72,5)]

for (cell_idx, cell_type) in veh_pairs:
    veh_pc[cell_idx] = cell_type

for (cell_idx, cell_type) in cno_pairs:
    cno_pc[cell_idx] = cell_type

file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
np.save(os.path.join(data_dir,file_name), veh_pc)


file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
np.save(os.path.join(data_dir,file_name), cno_pc)




mouse = 'DD2'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

#examples 
file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
veh_pc = np.load(os.path.join(data_dir, file_name))

file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
cno_pc = np.load(os.path.join(data_dir, file_name))

veh_pairs = [(7,5), (8,5), (21,4), (10,4), (14,5), (16,5),(24,5)]
cno_pairs = [(7,5), (8,5), (21,4), (14,4), (8,5), (12,4), (2,4)]

for (cell_idx, cell_type) in veh_pairs:
    veh_pc[cell_idx] = cell_type

for (cell_idx, cell_type) in cno_pairs:
    cno_pc[cell_idx] = cell_type

file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
np.save(os.path.join(data_dir,file_name), veh_pc)


file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
np.save(os.path.join(data_dir,file_name), cno_pc)



mouse = 'CalbV23'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

#examples 
file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
veh_pc = np.load(os.path.join(data_dir, file_name))

file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
cno_pc = np.load(os.path.join(data_dir, file_name))

veh_pairs = [(7,5),(8,5),(11,5),(28,0),(29,0),(30,5),(46,0),(49,0)]
cno_pairs = [(7,5),(8,5),(29,4),(30,5),(50,5),(54,1),(55,5)]

for (cell_idx, cell_type) in veh_pairs:
    veh_pc[cell_idx] = cell_type

for (cell_idx, cell_type) in cno_pairs:
    cno_pc[cell_idx] = cell_type

file_name =  mouse+'_veh_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
np.save(os.path.join(data_dir,file_name), veh_pc)


file_name =  mouse+'_CNO_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_CNO')
np.save(os.path.join(data_dir,file_name), cno_pc)


#__________________________________________________________________________
#|                                                                        |#
#|                   CREATE PD OF CELLS ONLY VEH CLASS                    |#
#|________________________________________________________________________|#
import seaborn as sns

mouse = 'CalbCharly11_concat'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

cell_type_list = list()

veh_activity_pre_list = list()
veh_activity_rot_list = list()
cno_activity_pre_list = list()
cno_activity_rot_list = list()


veh_metric_pre_list = list()
veh_metric_rot_list = list()
cno_metric_pre_list = list()
cno_metric_rot_list = list()
print(f"Working on mouse: {mouse}")
for case in ['veh', 'CNO']:
    file_name =  mouse+'_'+case+'_pc_dict.pkl'
    data_dir = os.path.join(base_dir, 'place_cells', mouse+'_'+case)
    animal_pc = load_pickle(data_dir, file_name)

    fnames = list(animal_pc.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    if 'veh' in case:
        file_name =  mouse+'_cellType.npy'
        data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_'+case)
        animal_cell_class = np.load(os.path.join(data_dir, file_name))

    for cell in range(animal_pc[fname_pre]['metric_val'].shape[0]):
        activity_pre = 20*np.sum(animal_pc[fname_pre]['total_firing_neurons'][cell,:])/animal_pc[fname_pre]['vel_sig'].shape[0]
        activity_rot = 20*np.sum(animal_pc[fname_rot]['total_firing_neurons'][cell,:])/animal_pc[fname_rot]['vel_sig'].shape[0]

        metric_pre = np.sum(animal_pc[fname_pre]['metric_val'][cell,:])
        metric_rot = np.sum(animal_pc[fname_rot]['metric_val'][cell,:])

        if 'veh' in case:
            veh_activity_pre_list.append(activity_pre)
            veh_activity_rot_list.append(activity_rot)
            veh_metric_pre_list.append(metric_pre)
            veh_metric_rot_list.append(metric_rot)

            cell_type = animal_cell_class[cell]
            cell_type = np.where([cell_type==0, np.logical_and(cell_type>0,cell_type<4), cell_type==4, cell_type==5])[0][0]
            cell_type = ['global', 'local', 'remapping', 'other'][cell_type]
            cell_type_list.append(cell_type)
        elif 'CNO' in case:
            cno_activity_pre_list.append(activity_pre)
            cno_activity_rot_list.append(activity_rot)
            cno_metric_pre_list.append(metric_pre)
            cno_metric_rot_list.append(metric_rot)



pd_cell_type = pd.DataFrame(data={'cell_type': cell_type_list,
                             'veh_activity_pre': veh_activity_pre_list,
                             'veh_activity_rot': veh_activity_rot_list,
                             'veh_metric_pre': veh_metric_pre_list,
                             'veh_metric_rot': veh_metric_rot_list,


                             'cno_activity_pre': cno_activity_pre_list,
                             'cno_activity_rot': cno_activity_rot_list,
                             'cno_metric_pre': cno_metric_pre_list,
                             'cno_metric_rot': cno_metric_rot_list

                             })

palette = {'global': '#ff218c', 'local': '#ffd800', 'remapping': '#21b1ff', 'other': '#cdd7d6'}

plt.figure()
ax = plt.subplot(1,3,1)
num_type = pd_cell_type.groupby('cell_type').count()['veh_activity_pre'].to_list()
num_type = [x/pd_cell_type.shape[0] for x in num_type]
ax.bar(['Global', 'Local', 'Remap', 'N/A'], num_type, color = [ '#ff218c', '#ffd800','#21b1ff', '#cdd7d6'])
temp_pd = pd_cell_type.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
ax = plt.subplot(2,3,2)
sns.scatterplot(data = temp_pd, x='veh_activity_pre', y = 'cno_activity_pre', hue='cell_type',palette = palette)
max_val = np.max([pd_cell_type['veh_activity_pre'].max(), pd_cell_type['cno_activity_pre'].max()])
ax.plot([0,max_val], [0,max_val], 'k--')
ax.get_legend().set_visible(False)
ax.set_aspect('equal')
temp_pd = pd_cell_type.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
ax = plt.subplot(2,3,3)
sns.scatterplot(data = temp_pd, x='veh_metric_pre', y = 'cno_metric_pre', hue='cell_type',palette = palette)
max_val = np.max([pd_cell_type['veh_metric_pre'].max(), pd_cell_type['cno_metric_pre'].max()])
ax.plot([0,max_val], [0,max_val], 'k--')
ax.get_legend().set_visible(False)
ax.set_aspect('equal')
ax = plt.subplot(2,3,5)
sns.scatterplot(data = temp_pd, x='veh_activity_rot', y = 'cno_activity_rot', hue='cell_type',palette = palette)
max_val = np.max([pd_cell_type['veh_activity_rot'].max(), pd_cell_type['cno_activity_rot'].max()])
ax.plot([0,max_val], [0,max_val], 'k--')
ax.get_legend().set_visible(False)
ax.set_aspect('equal')
temp_pd = pd_cell_type.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
ax = plt.subplot(2,3,6)
sns.scatterplot(data = temp_pd, x='veh_metric_rot', y = 'cno_metric_rot', hue='cell_type',palette = palette)
max_val = np.max([pd_cell_type['veh_metric_rot'].max(), pd_cell_type['cno_metric_rot'].max()])
ax.plot([0,max_val], [0,max_val], 'k--')
ax.get_legend().set_visible(False)
ax.set_aspect('equal')




plt.figure()
pd_cell_type['cno_vs_veh_activity'] = pd_cell_type['cno_activity_rot']-pd_cell_type['veh_activity_rot']
ax = plt.subplot(1,1,1)
sns.violinplot(data = pd_cell_type, x = 'cell_type', y='cno_vs_veh_activity',palette = palette, order=list(palette.keys()))

plt.figure()
temp_pd = pd_cell_type.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
ax = plt.subplot(1,1,1)
sns.scatterplot(data=temp_pd, x='cno_activity_pre', y='cno_activity_rot', hue='cell_type', palette = palette, ax=ax)
ax.plot([0,pd_cell_type['cno_activity_pre'].max()], [0,pd_cell_type['cno_activity_rot'].max()], 'r--')
ax.set_aspect('equal')
pd_cell_type['veh_activity_change'] = pd_cell_type['veh_activity_pre']-pd_cell_type['veh_activity_rot']
pd_cell_type['cno_activity_change'] = pd_cell_type['cno_activity_pre']-pd_cell_type['cno_activity_rot']
veh_std = np.std(np.abs(pd_cell_type['veh_activity_change'].to_list()))



plt.figure()
temp_pd = pd_cell_type.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
ax = plt.subplot(1,1,1)
sns.scatterplot(data=temp_pd, x='cno_activity_pre', y='cno_activity_rot', hue='cell_type', palette = palette, ax=ax)
max_val = np.max([pd_cell_type['cno_activity_pre'].max(), pd_cell_type['cno_activity_rot'].max()])
ax.plot([0,max_val], [0,max_val], 'k--')
ax.fill_between([0,max_val], [-veh_std,max_val-veh_std], [veh_std,max_val+veh_std], alpha=0.3, color='k', linewidth=0)
ax.set_aspect('equal')
ax.set_ylim([-0.01, max_val+0.01])
ax.set_xlim([-0.01, max_val+0.01])

num_cell_type = pd_cell_type.groupby('cell_type').count()['cno_activity_change'].to_list()
pd_cell_out = pd_cell_type[pd_cell_type['cno_activity_change']>veh_std]
num_out_type = pd_cell_out.groupby('cell_type').count()['cno_activity_change'].to_list()
perc = [num_out_type[idx]/num_cell_type[idx] for idx in range(len(num_out_type))]



pd_cell_type['veh_activity_change_perc'] = (pd_cell_type['veh_activity_rot']-pd_cell_type['veh_activity_pre'])/pd_cell_type['veh_activity_pre']
pd_cell_type['cno_activity_change_perc'] = (pd_cell_type['cno_activity_rot']-pd_cell_type['cno_activity_pre'])/pd_cell_type['cno_activity_pre']
veh_std = np.abs(pd_cell_type['veh_activity_change_perc'].to_list())
veh_std[veh_std==np.inf] = np.nan
veh_std = np.nanstd(veh_std)

num_cell_type = pd_cell_type.groupby('cell_type').count()['cno_activity_change_perc'].to_list()
pd_cell_out = pd_cell_type[pd_cell_type['cno_activity_change_perc']<-veh_std]
num_out_type = pd_cell_out.groupby('cell_type').count()['cno_activity_change_perc'].to_list()
perc = [num_out_type[idx]/num_cell_type[idx] for idx in range(len(num_out_type))]


pd_cell_type['veh_cno_activity_change'] = pd_cell_type['veh_activity_pre']-pd_cell_type['cno_activity_pre']
veh_std = np.std(np.abs(pd_cell_type['veh_cno_activity_change'].to_list()))


mouse = 'CalbCharly11_concat'
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'
case = 'CNO'

file_name =  mouse+'_cellType.npy'
data_dir = os.path.join(base_dir, 'functional_cells', mouse+'_veh')
animal_cell_type = np.load(os.path.join(data_dir, file_name))



file_name =  f"{mouse}_CNO_df_dict.pkl"
case_dir = os.path.join(base_dir, 'processed_data', mouse+'_CNO')
animal_CNO = load_pickle(case_dir,file_name)
fnames = list(animal_CNO.keys())
fname_pre_CNO = [fname for fname in fnames if 'lt' in fname][0]
fname_rot_CNO = [fname for fname in fnames if 'rot' in fname][0]



file_name =  f"{mouse}_veh_df_dict.pkl"
case_dir = os.path.join(base_dir, 'processed_data', mouse+'_veh')
animal_veh = load_pickle(case_dir,file_name)
fnames = list(animal_veh.keys())
fname_pre_veh = [fname for fname in fnames if 'lt' in fname][0]
fname_rot_veh = [fname for fname in fnames if 'rot' in fname][0]

    
traces_pre_veh = get_signal(animal_veh[fname_pre_veh], 'raw_traces')        
traces_rot_veh = get_signal(animal_veh[fname_rot_veh], 'raw_traces')        
traces_all_veh = np.concatenate((traces_pre_veh, traces_rot_veh),axis=0)
time_veh = np.arange(traces_all_veh.shape[0])/20

traces_pre_CNO = get_signal(animal_CNO[fname_pre_CNO], 'raw_traces')        
traces_rot_CNO = get_signal(animal_CNO[fname_rot_CNO], 'raw_traces') 
traces_all_CNO = np.concatenate((traces_pre_CNO, traces_rot_CNO),axis=0)
time_CNO = np.arange(traces_all_CNO.shape[0])/20

palette = {'global': '#ff218c', 'local': '#ffd800', 'remapping': '#21b1ff', 'other': '#cdd7d6'}

global_cells = np.where(animal_cell_type==0)[0]
local_cells = np.where(np.logical_and(animal_cell_type>0,animal_cell_type<4))[0]
remapping_cells = np.where(animal_cell_type==4)[0]
other_cells = np.where(animal_cell_type==5)[0]

fig, ax = plt.subplots(1,2)
idx = 0
for type_name, type_mat in [('global', global_cells), ('local', local_cells), ('remapping', remapping_cells), ('other', other_cells)]:
    for cell in range(np.min([10,len(type_mat)])):
        ax[0].plot(time_veh, traces_all_veh[:, type_mat[cell]]-idx*30, color=palette[type_name])

        ax[1].plot(time_CNO, traces_all_CNO[:, type_mat[cell]]-idx*30, color=palette[type_name])

        idx += 1
    idx+=1

ax[0].plot([traces_pre_veh.shape[0]/20]*2, [-(idx-2)*30-0.5, 1.5], 'r--')
ax[1].plot([traces_pre_CNO.shape[0]/20]*2, [-(idx-2)*30-0.5, 1.5], 'r--')

