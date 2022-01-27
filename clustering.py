# iris 데이터셋 kmean 분석 시각화
# 군집 분석 시각화

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from sklearn import datasets

# iris 데이터 셋 불러오기
iris = datasets.load_iris()
#
# data = iris.data[:,:2] # 다항에서는 첫번째 두 컬럼 만 선정
# target = iris.target

X , y = iris.data , iris.target

plt.scatter(X[:,0] , X[:,1], marker='o' , c = y , s = 100 , edgecolors= 'k' , linewidths= 2 )
plt.show()

range_k = [2,3,4,5,6]
# 생성할 군집의 수를 리스트로 정의

# 생성된 데이터를 기본 산점도로 출력
# k값에 따른 군집의 변화를 확인

# petal width , length 기준 군집 분석
# for k in range_k:
#     fig , (ax) = plt.subplots(1,1)
#     # 그래프 작성을 위한 초기화
#
#     # kmean 분석을 이용해서 군집화 시도
#     kmeans = KMeans(n_clusters=k , random_state=1)
#     predcits = kmeans.fit_predict(X)
#
#     # 실루엣 점수 출력
#     ss = silhouette_score(X, predcits)
#     print(k , '=>' , ss )
#
#
#
#     # 군집을 시각화 하기 위해 색상맵 정의
#     # 예측 결과에 따라 적절한 색상을 부여
#     colors = cm.nipy_spectral(predcits.astype(float) / k )
#
#     # 부여된 색상을 기본으로 산점도 그림
#     ax.scatter(X[:,0] , X[:,1], marker='.' , c = colors , s = 100 , edgecolors= 'k' )
#
#     # 그래프에 소제목 출력
#     plt.suptitle(('cluster => %d' % k), fontsize = 14 , fontweight = 'bold'  )
#
#     # 군집의 기준점을 표시
#     centers = kmeans.cluster_centers_
#
#     ax.scatter(centers[:,0] , centers[:,1] , marker = 'o' , c = 'white' , s = 200 , edgecolors ='k')
# plt.show()

# sepal width , length 기준 군집 분석
for k in range_k:
    fig , (ax) = plt.subplots(1,1)
    # 그래프 작성을 위한 초기화

    # kmean 분석을 이용해서 군집화 시도
    kmeans = KMeans(n_clusters=k , random_state=1)
    predcits = kmeans.fit_predict(X)

    # 실루엣 점수 출력
    ss = silhouette_score(X, predcits)
    print(k , '=>' , ss )



    # 군집을 시각화 하기 위해 색상맵 정의
    # 예측 결과에 따라 적절한 색상을 부여
    colors = cm.nipy_spectral(predcits.astype(float) / k )

    # 부여된 색상을 기본으로 산점도 그림
    ax.scatter(X[:,2] , X[:,3], marker='.' , c = colors , s = 100 , edgecolors= 'k' )

    # 그래프에 소제목 출력
    plt.suptitle(('cluster => %d' % k), fontsize = 14 , fontweight = 'bold'  )

    # 군집의 기준점을 표시
    centers = kmeans.cluster_centers_

    ax.scatter(centers[:,2] , centers[:,3] , marker = 'o' , c = 'white' , s = 200 , edgecolors ='k')
plt.show()

