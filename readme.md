
# 🎬 Anime Recommendation System

This project is an Anime Recommendation System that leverages a combination of content-based filtering and collaborative filtering to suggest anime based on user preferences. Built with Flask for the backend API and a sleek front end using Tailwind CSS and JavaScript. 🌟

## 📁 Project Structure

- **main.py**: 🧠 Script for training the anime recommendation model and generating required data files.
- **app.py**: 🚀 Flask API providing endpoints for anime recommendations.
- **index.html**: 🌐 Front-end interface designed with Tailwind CSS and JavaScript.
- **/**: 📂 Directory containing the raw anime dataset used for model training.
- **anime_dataset**: 📊 Explore `anime_dataset_extended_final.csv` for insights!
- **.gitignore**: 🗂️ Configuration to ignore generated files from version control.

## ⚙️ Setup Instructions

### Prerequisites

Ensure you have the following installed:

- 🐍 Python 3.7+
- 📦 pip (Python package manager)

### Install Dependencies

Run the following command to install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Run the Flask API

Start the Flask server with:

```bash
python app.py
```

The server will be live at `http://127.0.0.1:5000` 🌍

### Front-End Interface

Open `index.html` in your web browser to interact with the recommendation system. To avoid CORS issues, use a local HTTP server:

```bash
python -m http.server 8000
```

Then navigate to `http://localhost:8000/index.html` in your browser. 🚀

## 🧑‍💻 Usage

### New User Recommendations

1. **Preferred Genres**: Enter genres like "Action, Adventure, Romance".
2. **Liked Anime Titles**: Provide a list of anime you enjoyed, such as "Attack on Titan, Black Clover, Kimi no Todoke, Horimiya, Mashle, Vinland Saga, Berserk".
3. Click **Get Recommendations** to see your personalized anime list! 🎉

### Existing User Recommendations

⚠️ **Note:** The "Existing User" recommendation feature is currently under maintenance. Please use the "New User" option for now. 🔧

## 🚫 Files Ignored by Version Control

Files generated during the model training process and ignored by version control (`.gitignore`):

- `anime_model_data.csv`
- `collab_sim.npy`
- `tfidf_vectorizer.pkl`
- `content_sim.npy`
- `simulated_user_data.csv`

## 🚧 Current Limitations

- The **Existing User** recommendation feature is under maintenance. Please use the **New User** option.
- The app relies on the dataset in the `/` directory; model training is required before use.

## 🚀 To-Do

- Fix the existing user recommendation functionality.
- Enhance dataset quality and filtering for more accurate results.
- Add features like user authentication and personalized profiles.

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request to [Model Repo](https://github.com/HarshitKumar9030/anime_recommendation_model).

## 📜 License

This project is licensed under the MIT License.

## 📬 Contact

For any inquiries or support, please contact [Harshit Kumar](mailto:harshitkumar9030@gmail.com), reach out on [Instagram](https://instagram.com/_harshit.xd), or visit [leoncyriac.me](https://leoncyriac.me) 🌐.
