# first test for astrohack

# 3th party libraries
import csv
import matplotlib.pyplot as plt
import numpy as np



def open_csv(name):
    return np.loadtxt(open(name, "r"), delimiter=",")


def open_info(name):
    sample_list = []

    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            sample_list.append(row)
        
    return sample_list

    
def main():
    folder_name = 'Astrohack-master/Data/Sample/'
    folder_name2 = folder_name + 'Sample_Data/Sample_Data/SAMPLE/'
    filename = folder_name + 'sample.csv'

    sample_list = open_info(filename)

    sample_names = sample_list[0]
    sample_list = sample_list[1:]
    
    n_spam = len(sample_list)
    
    print(n_spam)
    

    
    for ind_im in range(n_spam):
        info_i = sample_list[ind_im]
        id_i = info_i[0]

        
        
        print(sample_names)
        print(info_i)
        
        print(id_i)
        
        name_im_g = folder_name2 + id_i + '-g.csv'
        # name_im_i =
        
        

        foo_b = open_csv(name_im_g)
        
        # print(foo_b)
        
        # with open(name_im_g, 'r') as csvfile:
        #     spamreader = csv.reader(csvfile, delimiter=',')
        #
        #     print(name_im_g)
        #
        #     for row in spamreader:
        #         print(row)
        #         foo_b.append(float(row))
        
        
        # foo_c = np.asarray(foo_b)
        
        print(np.min(foo_b))
        print(np.max(foo_b))

        # print(np.)
        
        print(np.shape(foo_b))

        plt.subplot(6, 9, ind_im+1)
        plt.title('{}'.format(info_i[1]))
        plt.imshow(foo_b, vmin=-1, vmax=1)

    plt.show()
    
if __name__ == '__main__':
    main()
