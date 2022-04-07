from torchvision import datasets 
import sys, random, datetime, torch, numpy as np

out = open('./statistics/'+sys.argv[1]+'-means.log','w')

def choose_init_mean(train_data,mean_vec,clusters,k):
    random.seed(datetime.datetime.now().timestamp())
    threshold = 60000 * random.random()
    out.write('The random number is '+str(threshold)+'\n')
    if k == 5:
        mean_vec_2={}
        for i in range(5):
            clusters[i] = []
        for i in range(60000):
            mean_vec_2[train_data[i][1]] = image2vec(train_data, i)
            if len(mean_vec_2.keys()) == 6:
                del mean_vec_2[train_data[i-1][1]]
                if i > threshold:
                    break
        idx = 0
        for key in mean_vec_2.keys():
            mean_vec[idx]=mean_vec_2[key]
            idx+=1
    elif k == '10':
        for i in range(10):
            clusters[i] = []
        for i in range(60000):
            mean_vec[train_data[i][1]] = image2vec(train_data, i)
            if len(mean_vec.keys()) == 10 and i > threshold:
                break
    elif k == '20':
        idx = 0
        mean_vec_2={}
        for i in range(20):
            clusters[i] = []
        for i in range(60000):
            mean_vec_2[train_data[i][1]] = image2vec(train_data, i)
            if len(mean_vec_2.keys()) == 10 and i > threshold/2:
                for key in mean_vec_2.keys():
                    mean_vec[idx]=mean_vec_2[key]
                    idx+=1
                mean_vec_2.clear()
                for j in range(i+1,60000):
                    mean_vec_2[train_data[j][1]] = image2vec(train_data, j)
                    if len(mean_vec_2.keys()) == 10 and j > threshold:
                        break
                break
        for key in mean_vec_2.keys():
            mean_vec[idx]=mean_vec_2[key]
            idx+=1

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
    converge = 0
    for i in range(epoch):
        out.write('\nepoch: '+str(i+1)+'\n')
        move = 0
        for key in clusters.keys():
            clusters[key].clear()
        for j in range(60000):
            scale = 40
            m = int((j+1)*40/60000)
            a = "*" * m
            b = "." * (scale - m)
            c = (m / scale) * 100
            d = ' '
            if i+1 < 10:
                d = '  '
            if i+1 > 99:
                d = ''
            print("\rrunning epoch: "+str(i+1)+d+"...... {:^3.0f}%[{}->{}]".format(c,a,b),end = "")
            vec = image2vec(train_data,j)
            distance = calculate_distance(vec,mean_vec[0])
            cluster_number = 0
            for k in range(1,len(mean_vec.keys())):
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
            print("\rrenewing mean vectors:   {:^3.0f}%[{}->{}]".format(c,a,b),end = "")
            new_k_mean = average_vec(clusters[k],train_data)
            move+=calculate_distance(new_k_mean,mean_vec[k])
            mean_vec[k] = new_k_mean
        print('')
        print("the mean vector move: "+str(move))
        out.write("the mean vector move: "+str(move)+'\n')
        print("main label of clusters: ",end='')
        out.write("main label of clusters: ")
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
            print("'"+str(main_ele)+"'",end='')
            print(':'+str(int(100*round(the_max/len(thelist),2)))+'%',end='  ')
            out.write("'"+str(main_ele)+"'"+':'+str(int(100*round(the_max/len(thelist),2)))+'%  ')
            for ele in thelist:
                if train_data[ele][1]!=main_ele:
                    count+=1
        print('\nacc: '+str((1-count/60000)*100)+'%')
        out.write('\nacc: '+str((1-count/60000)*100)+'%\n')
        np.save('./clusters/run3_20_epoch'+str(i+1)+'.npy',clusters)
        if move == 0:
            print("done after epoch "+str(i+1)+" !")
            out.write("done after epoch "+str(i+1)+" !\n")
            converge = 1
            break
        print()
    if converge == 0:
        print("not converged yet!")
        out.write("not converged yet!\n")


def main(argv):
    if len(sys.argv)!=2:
        print('usage: kmeans.py k\nk can be 5 or 10 or 20.')
        sys.exit()
    if sys.argv[1] != '5' and sys.argv[1] != '10' and sys.argv[1] != '20':
        print('usage: kmeans.py k\nk can be 5 or 10 or 20.')
        sys.exit()
    mean_vec={}
    clusters = {}
    train_data = datasets.MNIST(root = './data',train = True, download=False)
    choose_init_mean(train_data,mean_vec,clusters,sys.argv[1])
    train(2,train_data,mean_vec,clusters)
    out.close()

if __name__ == "__main__":
    main(sys.argv[1:])