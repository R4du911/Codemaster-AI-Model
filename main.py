import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

k_means = KMeans(n_clusters=2, random_state=52)
rf = RandomForestClassifier(max_depth=18, n_estimators=217)


app = Flask(__name__)


@app.before_request
def train_desk_model():
    # read data from csv
    filename_desks = "hackathon-schema.csv"
    read_data_desks = pd.read_csv(filename_desks)

    # map data to integer
    read_data_desks = read_data_desks.iloc[:, 2:]
    read_data_desks['firstHalf'] = read_data_desks['firstHalf'].astype(int)
    read_data_desks['secondHalf'] = read_data_desks['secondHalf'].astype(int)

    read_data_desks['date'] = pd.to_datetime(read_data_desks['date'], format='%d/%m/%Y')
    read_data_desks['month'] = read_data_desks['date'].dt.month
    read_data_desks['day'] = read_data_desks['date'].dt.day

    # feature engineering
    read_data_desks['first_part_of_week'] = (read_data_desks['date'].dt.dayofweek <= 3).astype(int)
    read_data_desks['second_part_of_week'] = (read_data_desks['date'].dt.dayofweek > 3).astype(int)
    read_data_desks['first_part_of_month'] = (read_data_desks['month'] <= 15).astype(int)
    read_data_desks['second_part_of_month'] = (read_data_desks['month'] > 15).astype(int)
    read_data_desks['occupancy_rate'] = ((read_data_desks['firstHalf'] + read_data_desks['secondHalf']) / 2)

    data_for_clustering = read_data_desks[['occupancy_rate', 'first_part_of_week', 'second_part_of_week',
                                           'first_part_of_month', 'second_part_of_month']]

    # train k_means
    k_means.fit(data_for_clustering)

    # put target of k_means in data
    read_data_desks['target'] = k_means.labels_
    data_for_classification = read_data_desks[['occupancy_rate', 'first_part_of_week', 'second_part_of_week',
                                               'first_part_of_month', 'second_part_of_month', 'target']]

    # split data for training and testing
    data_desks_train, data_desks_test, target_desks_train, target_desks_test = train_test_split(
        data_for_classification.iloc[:, :-1], data_for_classification['target'], test_size=0.2)

    # train random forest
    rf.fit(data_desks_train, target_desks_train)


@app.route("/desk-prediction", methods=['POST'])
def hello_world():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    prediction = rf.predict(input_df)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    with app.app_context():
        app.run()
