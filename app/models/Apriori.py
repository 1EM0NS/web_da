
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

class AprioriModel:
    def __init__(self, minSupport, minConf):
        self.minSupport = minSupport
        self.minConf = minConf

    def loadDataSet(self):
        return [['面包', '可乐', '麦片'], ['牛奶', '可乐'], ['牛奶', '面包', '麦片'], ['牛奶', '可乐'],
                ['面包', '鸡蛋', '麦片'], ['牛奶', '面包', '可乐'], ['牛奶', '面包', '鸡蛋', '麦片'],
                ['牛奶', '面包', '可乐'], ['面包', '可乐']]

    # return [[1, 2, 3], [4, 2], [4,1, 3], [4, 2], [1,5, 3], [4,1, 2], [4,1,5,3], [4,1, 2],[1,2]]

    # 获取候选1项集，dataSet为事务集。返回一个list，每个元素都是set集合
    def createC1(self, dataSet):
        C1 = []  # 元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）
        for transaction in dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()  # 这里排序是为了，生成新的候选集时可以直接认为两个n项候选集前面的部分相同
        # 因为除了候选1项集外其他的候选n项集都是以二维列表的形式存在，所以要将候选1项集的每一个元素都转化为一个单独的集合。
        return list(map(frozenset, C1))  # map(frozenset, C1)的语义是将C1由Python





    # 找出候选集中的频繁项集
    # dataSet为全部数据集，Ck为大小为k（包含k个元素）的候选项集，minSupport为设定的最小支持度
    def scanD(self,dataSet, Ck):
        ssCnt = {}  # 记录每个候选项的个数
        for tid in dataSet:
            for can in Ck:
                if can.issubset(tid):
                    ssCnt[can] = ssCnt.get(can, 0) + 1  # 计算每一个项集出现的频率
        numItems = float(len(dataSet))
        retList = []
        supportData = {}
        for key in ssCnt:
            support = ssCnt[key] / numItems
            if support >= self.minSupport:
                retList.insert(0, key)  # 将频繁项集插入返回列表的首部
                supportData[key] = support
        return retList, supportData  # retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度


    # 通过频繁项集列表Lk和项集个数k生成候选项集C(k+1)。
    def aprioriGen(sef,Lk, k):
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):
            for j in range(i + 1, lenLk):
                # 前k-1项相同时，才将两个集合合并，合并后才能生成k+1项
                L1 = list(Lk[i])[:k - 2];
                L2 = list(Lk[j])[:k - 2]  # 取出两个集合的前k-1个元素
                L1.sort();
                L2.sort()
                if L1 == L2:
                    retList.append(Lk[i] | Lk[j])
        return retList


    # 获取事务集中的所有的频繁项集
    # Ck表示项数为k的候选项集，最初的C1通过createC1()函数生成。Lk表示项数为k的频繁项集，supK为其支持度，Lk和supK由scanD()函数通过Ck计算而来。
    def apriori(self,dataSet):
        C1 = self.createC1(dataSet)  # 从事务集中获取候选1项集
        D = list(map(set, dataSet))  # 将事务集的每个元素转化为集合
        L1, supportData = self.scanD(D, C1,)  # 获取频繁1项集和对应的支持度
        L = [L1]  # L用来存储所有的频繁项集
        k = 2
        for k in range(2, 3):  # 一直迭代到项集数目过大而在事务集中不存在这种n项集
            Ck = self.aprioriGen(L[k - 2], k)  # 根据频繁项集生成新的候选项集。Ck表示项数为k的候选项集
            Lk, supK = self.scanD(D, Ck)  # Lk表示项数为k的频繁项集，supK为其支持度
            L.append(Lk);
            supportData.update(supK)  # 添加新频繁项集和他们的支持度
            k += 1
        return L, supportData, supK


    def draw_seaborn(self,suppData, n, list5):
        list1 = []
        list2 = []
        list3 = []
        for i in list5:
            list4 = []
            list1.append(i)
            list2.append(i)
            for j in list5:
                a = list(map(frozenset, [[i, j]]))[0]
                # print(suppData[a])
                if a in suppData:
                    list4.append(suppData[a])
                else:
                    list4.append(0.00)
            list3.append(list4)
        data = {}
        for num in range(len(n)):
            data[list5[num]] = list3[num]
        #         print(data)
        #         print(list5[num])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        df = pd.DataFrame(data, index=list5, columns=list5)
        sns.heatmap(df, annot=True, annot_kws={"size": 7})
        return list1,list2,list3


    def data_deal(self,L, suppData, list1):
        min = -1
        DataList = []
        for i in range(len(L[1])):
            for j in L[1][i]:
                if i > min:
                    min = i
                    List = []
                    List.append(suppData[L[1][i]])
                List.append(j)
            DataList.append(List)
        Datanum = {}
        m = 0
        for j in list1:
            Datanum[j] = L[0][m]
            m += 1
        return DataList, Datanum


    def confidence(self,DataList, Datanum, suppData):
        confidence_data = {}
        for num in range(len(DataList)):
            str1 = ''
            str2 = ''
            str1 = str(DataList[num][1]) + '->' + str(DataList[num][2])
            confidence_data[str1] = DataList[num][0] / suppData[Datanum[DataList[num][1]]]
            str2 = str(DataList[num][2]) + '->' + str(DataList[num][1])
            confidence_data[str2] = DataList[num][0] / suppData[Datanum[DataList[num][2]]]
        return confidence_data


    def Lift(self,DataList, confidence_data, Datanum, suppData):
        Lift_data = {}
        for num in range(len(DataList)):
            str1 = ''
            str2 = ''
            str1 = str(DataList[num][1]) + '->' + str(DataList[num][2])
            Lift_data[str1] = confidence_data[str1] / suppData[Datanum[DataList[num][2]]]
            str2 = str(DataList[num][2]) + '->' + str(DataList[num][1])
            Lift_data[str2] = confidence_data[str2] / suppData[Datanum[DataList[num][1]]]
        return Lift_data


    def limit(self,confidence_data, Lift_data):
        limit_data = []
        for k in confidence_data.keys():
            data = []
            if confidence_data[k] >= self.minConf:
                data.append(k)
                data.append(confidence_data[k])
                data.append(Lift_data[k])
                limit_data.append(data)
        return limit_data


    def datado(self,confidence_data):
        data = list(confidence_data.keys())
        #   print(data)
        return data


    def draw_seabornc(self,comfidence, list1, confidence_data):
        list3 = []
        for i in list1:
            list2 = []
            for j in list1:
                if i == j:
                    list2.append(1.00)
                else:
                    a = str(i + "->" + j)
                    if a in comfidence:
                        list2.append(confidence_data[a])
                    else:
                        list2.append(0.00)
            list3.append(list2)
        #    print(list3)
        data = {}
        for num in range(0, len(list1)):
            data[list1[num]] = list3[num]
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        df = pd.DataFrame(data, index=list1, columns=list1)
        return sns.heatmap(df, annot=True, annot_kws={"size": 7})
    def apriori_demo(self):

        dataSet = self.loadDataSet()  # 获取事务集。每个元素都是列表
        L, suppData, supK = self.apriori(dataSet)
        print(supK)
        list1 = []
        for i in L[0]:
            for j in i:
                if j not in list1:
                    list1.append(j)
        DataList, Datanum = self.data_deal(L, suppData, list1)
        confidence_data = self.confidence(DataList, Datanum, suppData)
        comfidence = self.datado(confidence_data)
        self.draw_seabornc(comfidence,list1, confidence_data)
        # plt.show()
        Lift_data = self.Lift(DataList, confidence_data, Datanum, suppData)
        #     print(Lift_data)
        limit_data = self.limit(confidence_data, Lift_data)
        print('\n')
        for i in limit_data:
            print(i)
        x, y, data = self.draw_seaborn(suppData, L[0], list1)
        # plt.show()
        return x, y, data


if __name__ == '__main__':
    model = AprioriModel(minSupport=0.2, minConf=0.2)
    dataSet = model.loadDataSet()  # 获取事务集。每个元素都是列表
    # print(createC1(dataSet))  # 获取候选1项集。每个元素都是集合
    # D = list(map(set, dataSet))  # 转化事务集的形式，每个元素都转化为集合。
    # L1, suppDat = scanD(D, C1, 0.5)
    # print(L1,suppDat)

    L, suppData, supK =model.apriori(dataSet)

    # print(supK)
    list1 = []
    for i in L[0]:
        for j in i:
            if j not in list1:
                list1.append(j)

    DataList, Datanum = model.data_deal(L, suppData, list1)

    confidence_data = model.confidence(DataList, Datanum, suppData)
    comfidence = model.datado(confidence_data)
    model.draw_seabornc(comfidence,list1, confidence_data)
    Lift_data = model.Lift(DataList, confidence_data, Datanum, suppData)
    #     print(Lift_data)
    limit_data = model.limit(confidence_data, Lift_data)
    print('\n')
    for i in limit_data:
        print(i)
    # x,y,data = model.draw_seaborn(suppData, L[0], list1)
    # plt.show()

