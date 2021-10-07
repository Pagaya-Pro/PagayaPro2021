# lubbock_forms= (
#     glob.glob(
#         os.path.join(ASSIGNMENT1_DATA,"offers/*lubbock*")
#     )
# )
# lubbock_df= pd.read_csv(lubbock_forms[0])
# lubbock_df['file_num'] = 0
# for num in range(1,len(lubbock_forms)):
#     lubbock_df = pd.concat([lubbock_df, pd.read_csv(raleigh_forms[num])], ignore_index=True)
# lubbock_df = lubbock_df.sort_values('offer_date', ascending=False).drop_duplicates('id').reset_index(drop=True)

import numpy as np

print(np.random.choice(np.arange(10), size=(100, 2)))