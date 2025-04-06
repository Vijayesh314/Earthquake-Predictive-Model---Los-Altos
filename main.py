import requests
import pandas as pd
import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import webbrowser

def scrape_earthquake_data():
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(url)
        return data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def preprocess_data(df):
    df = df[['time', 'latitude', 'longitude', 'depth', 'mag', 'place']].copy()
    df.columns = ['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Location']
    df['Time'] = pd.to_datetime(df['Time'])

    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Day'] = df['Time'].dt.day
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Second'] = df['Time'].dt.second

    df = df.dropna()
    return df

def plot_magnitude_distribution(df):
    plt.figure(figsize=(14, 8))
    sns.histplot(df['Magnitude'], bins=30, color='cornflowerblue', edgecolor='black', linewidth=1.2)

    mean_magnitude = df['Magnitude'].mean()
    plt.axvline(mean_magnitude, color='red', linestyle='--', linewidth=2)
    plt.text(mean_magnitude + 0.1, plt.ylim()[1] * 0.9, f'Mean: {mean_magnitude:.2f}', color='red', fontsize=14, fontweight='bold')

    median_magnitude = df['Magnitude'].median()
    plt.axvline(median_magnitude, color='green', linestyle='--', linewidth=2)
    plt.text(median_magnitude + 0.1, plt.ylim()[1] * 0.8, f'Median: {median_magnitude:.2f}', color='green', fontsize=14, fontweight='bold')

    plt.title('Distribution of Earthquake Magnitudes', fontsize=20, fontweight='bold', color='navy')
    plt.xlabel('Magnitude', fontsize=16, fontweight='bold')
    plt.ylabel('Frequency', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

def plot_enhanced_geo_map(df):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        magnitude = row['Magnitude']
        if magnitude < 3.0:
            color = 'blue'
        elif 3.0 <= magnitude < 5.0:
            color = 'green'
        elif 5.0 <= magnitude < 7.0:
            color = 'orange'
        else:
            color = 'red'

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=(f"Location: {row['Location']}<br>"
                   f"Time: {row['Time']}<br>"
                   f"Magnitude: {row['Magnitude']}<br>"
                   f"Depth: {row['Depth']} km"),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    return m

def train_model(X_train, y_train):
    """
    Trains a Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using test data and visualizes results.
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✔️ Accuracy Score: {accuracy:.4f}")
    print(f"🧮 F1 Score: {f1:.4f}")

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Significant', 'Significant'], 
                yticklabels=['Not Significant', 'Significant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Scrape data
    earthquake_data = scrape_earthquake_data()

    if earthquake_data is None:
        return

    # Preprocess
    processed_data = preprocess_data(earthquake_data)
    plot_magnitude_distribution(processed_data)

    # Map Visualization
    enhanced_earthquake_map = plot_enhanced_geo_map(processed_data)
    enhanced_earthquake_map.save("enhanced_earthquake_map.html")
    webbrowser.open("enhanced_earthquake_map.html")

if __name__ == "__main__":
    main()