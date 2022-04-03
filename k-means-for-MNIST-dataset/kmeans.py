from torchvision import datasets 
import torch, numpy as np

def choose_init_mean(train_data,mean_vec,clusters):
    for i in range(60000):
        if train_data[i][1] not in mean_vec.keys():
            mean_vec[train_data[i][1]] = image2vec(train_data, i)
            clusters[train_data[i][1]] = []
        if len(mean_vec.keys()) == 10:
            break

def image2vec(train_data,idx):
    pic = train_data[idx][0]
    pic14 = pic.resize((14,14))
    pic7 = pic.resize((7,7))
    vec = []
    for i in range(7):
        for j in range(7):
            vec.append(6*pic7.getpixel((i,j))+0.0)
    for i in range(14):
        for j in range(14):
            vec.append(pic14.getpixel((i,j))+0.0)
    return vec

def calculate_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    return torch.dist(torch.from_numpy(a),torch.from_numpy(b),2)

def average_vec(vecs,train_data):
    num = len(vecs)
    output = np.zeros(245)
    for ele in vecs:
        output+=np.array(image2vec(train_data,ele))
    output/=num
    return output.tolist()

def train(epoch,train_data,mean_vec,clusters):
    for i in range(epoch):
        print()
        move = 0
        for key in clusters.keys():
            clusters[key].clear()
        for j in range(60000):
            scale = 40
            m = int((j+1)*40/60000)
            a = "*" * m
            b = "." * (scale - m)
            c = (m / scale) * 100
            d = ''
            if i+1 < 10:
                d = ' '
            print("\rrunning epoch: "+str(i+1)+d+"...... {:^3.0f}%[{}->{}]".format(c,a,b),end = "")
            vec = image2vec(train_data,j)
            distance = calculate_distance(vec,mean_vec[0])
            cluster_number = 0
            for k in range(1,10):
                new_distance = calculate_distance(vec,mean_vec[k])
                if new_distance<distance:
                    distance = new_distance
                    cluster_number = k
            clusters[cluster_number].append(j)
        print('')
        for k in range(len(mean_vec.keys())):
            a = "*" * (k+1) * int(40/len(mean_vec.keys()))
            b = "." * (40 - (k+1) * int(40/len(mean_vec.keys())))
            c = ((k+1) / len(mean_vec.keys())) * 100
            print("\rrenewing mean vectors:  {:^3.0f}%[{}->{}]".format(c,a,b),end = "")
            new_k_mean = average_vec(clusters[k],train_data)
            move+=calculate_distance(new_k_mean,mean_vec[k])
            mean_vec[k] = new_k_mean
        print('')
        print("the mean vector move: "+str(move))
        print("main label of clusters: ",end='')
        count = 0
        for j in range(len(mean_vec.keys())):
            thelist = clusters[j]
            count_max = {}
            for ele in thelist:
                if train_data[ele][1] in count_max.keys():
                    count_max[train_data[ele][1]]+=1
                else:
                    count_max[train_data[ele][1]]=1
            the_max = 0
            main_ele = -1
            for key in count_max.keys():
                if count_max[key] > the_max:
                    the_max = count_max[key]
                    main_ele = key
            print(main_ele,end=' ')
            for ele in thelist:
                if train_data[ele][1]!=main_ele:
                    count+=1
        print('\nacc: '+str((1-count/60000)*100)+'%')
        np.save('./clusters/strategy1_epoch'+str(i)+'.npy',clusters)
    print("done after epoch "+str(epoch)+" !")


def main():
    mean_vec={}
    clusters = {}
    train_data = datasets.MNIST(root = './data',train = True, download=False)
    choose_init_mean(train_data,mean_vec,clusters)
    train(99,train_data,mean_vec,clusters)

if __name__ == "__main__":
    main()