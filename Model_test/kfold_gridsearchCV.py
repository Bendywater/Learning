# 做一个项目，使用wine数据集一个多分类任务上训练一个随机森林模型，使用网格搜索找到一组最佳超参数
# 用stratifiedKFold保证每一折交叉验证，类别分布保持一致
# 输出模型在交叉验证下的最优准确率
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

# 加载数据
X, y = load_wine(return_X_y=True)

print(X.shape,y.shape)
# 定义参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# 使用 StratifiedKFold 保证每一折中类分布一致
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化模型与网格搜索器
clf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(clf, param_grid, cv=skf, scoring='accuracy')

# 训练模型
grid.fit(X, y)

# 输出结果
print("最优参数:", grid.best_params_)
print("最优交叉验证准确率:", grid.best_score_)