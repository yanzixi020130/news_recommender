import os
import pickle
import pandas as pd
import warnings
import collections
from tqdm import tqdm

warnings.filterwarnings('ignore')

from data_processing import get_all_click_df
from utils import get_user_item_time, get_item_topk_click
from models.recall_baseline import itemcf_sim, item_based_recommend, generate_user_recall_dict
from submission import submit
from data_processing import get_item_info_df
from utils import  get_item_info_dict

data_path = './data_raw/'
save_path = './results/'

# 用于生成线上提交的流程（train + testA）
def main(sim_version='submit_v1', recall_version='submit_v1', use_cache=True):
    print("🚀 main() started for online submission generation (train + testA)")  # main() 启动，用于生成线上提交结果（train + testA）

    # Step 1：加载点击数据（包含 train 和 testA 用户）
    all_click_df = get_all_click_df(data_path=data_path, offline=False)
    print(f"✅ Data loaded, total click records: {len(all_click_df)}")  # 数据加载完成，共 ... 条点击记录

    # Step 2：计算物品相似度矩阵（版本化+缓存机制）
    print("📌 [Step 2] Computing item similarity matrix...")  # 计算物品相似度矩阵

    item_info_df = get_item_info_df(data_path)
    item_created_time_dict = get_item_info_dict(item_info_df)[2]

    sim_cache_file = os.path.join('cache', f'itemcf_sim_{sim_version}.pkl')
    i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, save_path=sim_cache_file, use_cache=use_cache)
    print("✅ Similarity matrix ready")  # 相似度矩阵准备完成

    # Step 3：初始化用户召回字典
    print("📌 [Step 3] Initializing user recall dictionary...")  # 初始化用户召回字典
    user_recall_items_dict = collections.defaultdict(dict)

    # Step 4：用户-文章时间序列
    print("📌 [Step 4] Computing user-article click time sequence...")  # 计算用户-文章点击时间序列
    user_item_time_dict = get_user_item_time(all_click_df)
    print(f"✅ User-article time sequence computed, users: {len(user_item_time_dict)}")  # 用户-文章时间序列计算完成

    # Step 5~8：召回参数配置
    sim_item_topk = 10
    recall_item_num = 10
    print(f"📌 [Step 5] Similar items count: {sim_item_topk}")  # 相似物品数量
    print(f"📌 [Step 6] Recall items per user: {recall_item_num}")  # 每个用户召回数量

    print("📌 [Step 7] Computing most popular articles...")  # 计算最热文章
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    print(f"✅ Popular articles computed, top 5: {item_topk_click[:5]}")  # 热门文章计算完成，前5篇

    # Step 8：生成用户推荐列表（只对 testA 用户）
    print("📌 [Step 8] Loading test users and generating recommendations...")  # 加载测试集用户并生成推荐列表

    # 读取 testA 用户
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users_df = pd.DataFrame({'user_id': tst_click['user_id'].unique()})
    print(f"✅ Test users loaded, total users: {len(tst_users_df)}")  # 测试集用户加载完成

    # 生成推荐缓存路径（支持版本命名）
    recall_cache_file = os.path.join('cache', f'user_recall_items_{recall_version}.pkl')

    # 生成推荐字典（只为 testA 用户生成）
    user_recall_items_dict = generate_user_recall_dict(
        val_df=tst_users_df,
        user_item_time_dict=get_user_item_time(all_click_df),
        i2i_sim=i2i_sim,
        sim_item_topk=10,
        recall_item_num=10,
        item_topk_click=item_topk_click,
        save_path=recall_cache_file,
        use_cache=use_cache
    )

    # Step 9：字典 → DataFrame
    print("📌 [Step 9] Converting recall dict to DataFrame...")  # 推荐字典转 DataFrame
    user_item_score_list = []
    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])
    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    print(f"✅ Recommendation DataFrame created, records: {len(recall_df)}")  # 推荐 DataFrame 生成完成

    # Step 10：生成提交文件
    print("📌 [Step 10] Generating submission file...")  # 生成提交文件
    submit(recall_df, save_path, topk=5, model_name=f'itemcf_{recall_version}')
    print("✅ Submission file generated!")  # 提交文件生成完成！

    print("🎉 main() completed!")  # main() 运行完成！

if __name__ == "__main__":
    main(sim_version='baseline_v1', recall_version='baseline_v1', use_cache=False)
