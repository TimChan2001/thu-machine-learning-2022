from torchvision import datasets 
import sys, random, datetime, torch, numpy as np
import matplotlib.pyplot as plt

out = open('./statistics/'+sys.argv[1]+'-means.log','w')
samples = (np.load('./tsne/samples.npy',allow_pickle=True)).tolist()
points = np.load('./tsne/points.npy',allow_pickle=True)

def choose_init_mean(train_data,mean_vec,clusters,k): # 选择初始cluster center
    random.seed(datetime.datetime.now().timestamp())
    threshold = 60000 * random.random() # 以时间为种子生成一个随机数
    out.write('The random number is '+str(threshold)+'\n')

    if k == '5': # k为5时的策略
        mean_vec_2={}
        for i in range(5):
            clusters[i] = []
        for i in range(60000):
            mean_vec_2[train_data[i][1]] = image2vec(train_data, i) # 每种label中选一个图片作为cluster center，保证不同的cluster center的label不同
            if len(mean_vec_2.keys()) == 6: # 当选择了6个图片作为cluster center时，去掉当前选择的cluster center的上一个cluster center，这样可以使每次确定选择的label随着随机数不同而变化
                del mean_vec_2[train_data[i-1][1]]
                if i > threshold: # 当i超过随机数时确定cluster center
                    break
        idx = 0
        for key in mean_vec_2.keys(): # 将cluster center按照0到4编号
            mean_vec[idx]=mean_vec_2[key]
            idx+=1

    elif k == '10': # k为10时的策略
        for i in range(10):
            clusters[i] = []
        for i in range(60000):
            mean_vec[train_data[i][1]] = image2vec(train_data, i) # 每种label中选一个图片作为cluster center，保证不同的cluster center的label不同
            if len(mean_vec.keys()) == 10 and i > threshold: # 当i超过随机数且选够10个时确定cluster center
                break
    
    elif k == '20': # k为20时的策略
        idx = 0
        mean_vec_2={}
        for i in range(20):
            clusters[i] = []
        for i in range(60000):
            mean_vec_2[train_data[i][1]] = image2vec(train_data, i) # 先选10个，每种label中选一个图片作为cluster center，保证不同的cluster center的label不同
            if len(mean_vec_2.keys()) == 10 and i > threshold/2: # 选够10个且i大于随机数的一半时开始选后10个
                for key in mean_vec_2.keys():
                    mean_vec[idx]=mean_vec_2[key] # 给选定的cluster centers编号
                    idx+=1
                mean_vec_2.clear()
                for j in range(i+1,60000):
                    mean_vec_2[train_data[j][1]] = image2vec(train_data, j) # 选后10个，每种label中选一个图片作为cluster center，保证不同的cluster center的label不同
                    if len(mean_vec_2.keys()) == 10 and j > threshold:
                        break
                break
        for key in mean_vec_2.keys():
            mean_vec[idx]=mean_vec_2[key] # 给选定的cluster centers编号
            idx+=1

def image2vec(train_data,idx): # 将图片向量化，这部分的处理详见实验报告
    pic = train_data[idx][0]
    pic14 = pic.resize((14,14)) # 适当模糊化降维
    pic7 = pic.resize((7,7)) # 适当模糊化降维
    vec = []
    for i in range(7):
        for j in range(7):
            vec.append(6*pic7.getpixel((i,j))+0.0) # 系数6经过不断修正得到
    for i in range(14):
        for j in range(14):
            vec.append(pic14.getpixel((i,j))+0.0)
    return vec

def calculate_distance(a,b): # 计算欧氏距离
    a = np.array(a)
    b = np.array(b)
    return torch.dist(torch.from_numpy(a),torch.from_numpy(b),1)

def average_vec(vecs,train_data): # 计算vecs中所有向量的均值作为新的中心
    num = len(vecs)
    output = np.zeros(245)
    for ele in vecs:
        output+=np.array(image2vec(train_data,ele))
    output/=num
    return output.tolist()

def visualize(clusters,run): # 可视化
    print("visualizing......")
    colors = []
    for ele in samples: # samples为sample用来可视化的图片编号，points为这些图片降到2维后的向量，由tsne.py得到
        for i in range(len(clusters)):
            if ele in clusters[i]:
                colors.append(i) # 根据聚类确定颜色
                break
    plt.figure(figsize=(7,7))
    plt.scatter(points[:, 0], points[:, 1],c=colors, cmap=plt.cm.Spectral)
    plt.savefig('./pics/visualization_'+sys.argv[1]+'-means_'+str(run)+'.png') # 本次聚类的可视化结果，ground truth可见./tsne/visualization_ground_truth.png
    print('check the picture ./pics/visualization_'+sys.argv[1]+'-means_'+str(run)+'.png')
    # plt.show()


def train(epoch,train_data,mean_vec,clusters): # k-means训练过程
    converge = 0 # 用来判断是否完全收敛
    for i in range(epoch):
        out.write('\nepoch: '+str(i+1)+'\n')
        move = 0 # 用来表示所有cluster center的移动，当move==0时完全收敛，converge = 1
        for key in clusters.keys():
            clusters[key].clear() # 清空上一次聚类的结果
        for j in range(60000): # 遍历所有样本
            scale = 40 ### 进度条
            m = int((j+1)*40/60000)
            a = "*" * m
            b = "." * (scale - m)
            c = (m / scale) * 100
            d = ' '
            if i+1 < 10:
                d = '  '
            if i+1 > 99:
                d = ''
            print("\rrunning epoch: "+str(i+1)+d+"...... {:^3.0f}%[{}->{}]".format(c,a,b),end = "") ### 进度条
            vec = image2vec(train_data,j) # 向量化
            distance = calculate_distance(vec,mean_vec[0]) # 初始化到中心距离
            cluster_number = 0
            for k in range(1,len(mean_vec.keys())):
                new_distance = calculate_distance(vec,mean_vec[k])
                if new_distance<distance: # 更新到中心距离，确定中心编号
                    distance = new_distance
                    cluster_number = k
            clusters[cluster_number].append(j) # 归类
        print('')
        for k in range(len(mean_vec.keys())):
            a = "*" * (k+1) * int(40/len(mean_vec.keys())) ### 进度条
            b = "." * (40 - (k+1) * int(40/len(mean_vec.keys())))
            c = ((k+1) / len(mean_vec.keys())) * 100
            print("\rrenewing mean vectors:   {:^3.0f}%[{}->{}]".format(c,a,b),end = "") ### 进度条
            new_k_mean = average_vec(clusters[k],train_data)
            move+=calculate_distance(new_k_mean,mean_vec[k]) # 计算中心的移动
            mean_vec[k] = new_k_mean
        print('')
        print("the mean vector move: "+str(move))
        out.write("the mean vector move: "+str(move)+'\n')
        print("main label of clusters: ",end='')
        out.write("main label of clusters: ")
        count = 0
        for j in range(len(mean_vec.keys())): # 计算每个类中最多的label
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
            print(':'+str(int(100*round(the_max/len(thelist),2)))+'%',end='  ') # 打出每个类中的准确率
            out.write("'"+str(main_ele)+"'"+':'+str(int(100*round(the_max/len(thelist),2)))+'%  ')
            for ele in thelist:
                if train_data[ele][1]!=main_ele:
                    count+=1
        print('\nacc: '+str((1-count/60000)*100)+'%') # 总准确率
        out.write('\nacc: '+str((1-count/60000)*100)+'%\n')
        if move == 0: # 完全收敛，结束
            print("done after epoch "+str(i+1)+" !")
            out.write("done after epoch "+str(i+1)+" !\n")
            converge = 1
            break
        if i%5 == 0: # 每五轮打一张图片
            visualize(clusters,i)
        print()
    if converge == 0: # 200轮训完没收敛
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
    train(200,train_data,mean_vec,clusters)
    out.close()

if __name__ == "__main__":
    main(sys.argv[1:])