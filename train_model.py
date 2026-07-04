import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('new_crop.csv',index_col=0)
df = df.dropna()

X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']

X_col=X.columns.to_list()
object_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
oh_train_arr = oh_encoder.fit_transform(X_train[object_cols])
oh_test_arr = oh_encoder.transform(X_test[object_cols])

oh_cols_train = pd.DataFrame(oh_train_arr, columns=oh_encoder.get_feature_names_out(object_cols), index=X_train.index)
oh_cols_test = pd.DataFrame(oh_test_arr, columns=oh_encoder.get_feature_names_out(object_cols), index=X_test.index)

X_train_final = pd.concat([X_train[num_cols], oh_cols_train], axis=1)
X_test_final = pd.concat([X_test[num_cols], oh_cols_test], axis=1)

X_test_final.to_csv('x_test.csv', index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_predicted = model.predict(X_test_scaled)
print(f"mean_squared_error: {mean_squared_error(y_test, y_predicted)}")

l=['East','Silt','Maize',609.7987,38.2658,True,False,'Cloudy',127]
data=pd.DataFrame([l],columns=X_col)
print(data)
print(data[object_cols])
oh_unseen_arr = oh_encoder.transform(data[object_cols])
oh_cols_unseen=pd.DataFrame(oh_unseen_arr,columns=oh_encoder.get_feature_names_out(object_cols),index=data.index)
print(oh_cols_unseen)
data_final=pd.concat([data[num_cols],oh_cols_unseen],axis=1)
scaled_data=scaler.transform(data_final)
print(f"predicted yield: {model.predict(scaled_data)}")


def predict_from_form(form_data):
    l=form_data
    data=pd.DataFrame([l],columns=X_col)
    print(data)
    print(data[object_cols])


    oh_unseen_arr = oh_encoder.transform(data[object_cols])
    oh_cols_unseen=pd.DataFrame(oh_unseen_arr,columns=oh_encoder.get_feature_names_out(object_cols),index=data.index)
    print(oh_cols_unseen)
    data_final=pd.concat([data[num_cols],oh_cols_unseen],axis=1)
    scaled_data=scaler.transform(data_final)
    print(f"predicted yield: {model.predict(scaled_data)}")
    return model.predict(scaled_data)
    

