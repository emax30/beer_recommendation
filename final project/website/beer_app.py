from flask import Flask, request, render_template, redirect, url_for, jsonify
from functions import *
from surprise import KNNBasic
from surprise.dump import load
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_beers', methods=['POST'])
def process_beers():
    beer_list = request.form['beer_list']
    beers = [beer.strip() for beer in beer_list.split(',')] # get user's favorite beers 
    province = request.form['province'] # and chosen province(s)

    if len(beers) < 3:
        return redirect(url_for('home')) # user will be prompted to enter at least 3 beers
    
    new_ratings_df, new_ratings_beers, new_ratings_styles = get_beers(beers) # generate a df, fuzzy-match beer names and find beer styles
    knn_model = load('website\knn_model')
    pred = rating_predictions(new_ratings_df, knn_model) # predict user's ratings for all other beers

    if len(pred) > 50: 
        if province == 'default': 
            output_beers, output_info, output_breweries, output_addresses = beer_recommendations(pred, new_ratings_styles)
            final_result = list(zip(output_beers, output_info, output_breweries, output_addresses))
        else:
            output_beers, output_info, output_breweries, output_addresses = prov_beer_recommendations(pred, new_ratings_styles, province)
            final_result = list(zip(output_beers, output_info, output_breweries, output_addresses))
    else:
        try:
            output_beers, output_info, output_breweries, output_addresses = alternative_recommendations(new_ratings_beers)
            final_result = list(zip(output_beers, output_info, output_breweries, output_addresses))
        except:
            final_result = None
    return render_template('results.html', result=final_result)     

@app.route('/results', methods=['GET'])
def results():
        return render_template('results.html')




