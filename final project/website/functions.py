import pandas as pd
import numpy as np
from surprise import Dataset, Reader, accuracy, KNNBasic
from surprise.model_selection import train_test_split
from surprise.dump import load
from fuzzywuzzy import process

breweries_df = pd.read_csv('./website/data/breweries_filtered.csv') # table of breweries with addresses
ratings_df = pd.read_csv('./website/data/full_beer_ratings.csv', dtype={'user_id': str, 'beer_id': str}) # table of user ratings with beer info
final_ratings = pd.read_csv('./website/data/num_beer_ratings.csv', dtype={'user_id': str, 'beer_id': str}) # shortened table of user ratings without beer info 
beer_df = pd.read_csv('./website/data/beer_info_final.csv')    # data on different beers used for recommendations

def get_beers(responses):
    """
    Accepts user's input (their favorite beers), finds the names using fuzzy matching and looks up beer styles and ids.
    Returns a df that can be used for the KNN model as well as beer names and ids as they appear in the beer_df.
    """
    
    beer_names = []
    beer_ids = []
    beer_styles = []
    choices = ratings_df['beer'].unique()
    for resp in responses:
        beer_names.append(process.extractOne(resp, choices)[0]) # fuzzy matching for each of the beers the user entered
    for beer in beer_names:
        beer_ids.append(ratings_df[ratings_df['beer']==beer]['beer_id'].iloc[0]) # getting beer ids
        if beer in list(beer_df['beer']):
            beer_styles.append(beer_df[beer_df['beer']==beer]['style'].iloc[0]) # getting beer styles
        
    user_ratings = [('1', beer_id, 5) for beer_id in beer_ids]
    df = pd.DataFrame(user_ratings, columns=['user_id', 'beer_id', 'user_rating']) # creating a df
    
    return (df, beer_names, beer_styles)

def rating_predictions(df, model):
    """
    Accepts a dataframe with user id, their favorite beer ids and their ratings (all set to 5).
    Returns a list of beers with their predicted ratings for this user.
    """
    updated_df = pd.concat([final_ratings, df])
    reader = Reader(rating_scale=(0, 5))
    updated_data = Dataset.load_from_df(updated_df, reader)
    new_train, new_test = train_test_split(updated_data, test_size=0.2, random_state=50)
    alg = model[1]
    alg.fit(new_train)

    beers_to_predict = updated_df[~updated_df['beer_id'].isin(df['beer_id'])]['beer_id'] # beers not rated by the user
    new_predictions = [alg.predict('1', beer_id) for beer_id in beers_to_predict] # generating predicted ratings for these beers

    final_predictions = []
    already_checked = set()

    # Making sure there are no duplicates or beers for which prediction was impossible (e.g. not enough neighbors)
    for prediction in new_predictions:
        if not prediction.details['was_impossible'] and prediction.iid not in already_checked:
            final_predictions.append(prediction)
            already_checked.add(prediction.iid)

    return final_predictions

def beer_recommendations(predictions, users_beer_styles):
    """
    Accepts a list of rating predictions produced by a surprise model and beer styles for user's favorite beers.
    Returns beers with the highest predicted rating that match user's favorite beer styles as recommendations. Also returns some 
    extra info about each beer (style, brewery and its address).
    """
    NUM_RECS = 5 # number of recommendations to be returned
    
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True) # sort the ratings from highest to lowest
    rec_list = []

    # Getting top 5 recommended beers from the list of predictions.
    for rec in recommendations:
        rec_beer = ratings_df[ratings_df['beer_id'] == rec.iid]['beer'].iloc[0] # find the beer based on its id
        current = beer_df[beer_df['beer'] == rec_beer] # look up the beer in the beer_df
        status = current['status'].iloc[0] if len(current) == 1 else None
        style = current['style'].iloc[0] if len(current) == 1 else None

        # If it was possible to identify user's favorite beer styles, take them into account
        if len(users_beer_styles) != 0:
            if rec_beer in list(beer_df['beer']) and style in users_beer_styles:
                rec_list.append(rec_beer)
                if len(rec_list) >= NUM_RECS:
                    break
        # Otherwise just return top recommendations regardless of style
        else:
            if rec_beer in list(beer_df['beer']):
                rec_list.append(rec_beer)
                if len(rec_list) >= NUM_RECS:
                    break

    # If beer style, status, brewery and brewery address are known, return these to the user.
    output_list = []
    info_list = []
    brewery_list = []
    addresses_list = []
    for rec in rec_list:
        current = beer_df[beer_df['beer'] == rec]
        if len(current) != 0:
            style = current['style'].iloc[0]
            status = current['status'].iloc[0]
            brewery = current['brewery'].iloc[0]
            if status == 'regular': # beer status is either regular or seasonal (e.g. summer seasonal, winter seasonal etc.)
                output_list.append(rec)
                info_list.append(f'It is a {style} brewed by {brewery}.')
            else:
                output_list.append(rec)
                info_list.append(f'It is a {style}, a {status} beer brewed by {brewery}.')

            # if the brewery is not a client brewer or a commercial brewery, provide their address
            if brewery in breweries_df['brewery'].unique():
                brewery_type = breweries_df[breweries_df['brewery'] == brewery]['type'].iloc[0]
                brewery_address = breweries_df[breweries_df['brewery'] == brewery]['address'].iloc[0]
                visitable = ['Microbrewery', 'Brewpub/Brewery', 'Brewpub']
                if brewery_type in visitable:
                    brewery_list.append(f'They are a {brewery_type} and you can visit them at ')
                    addresses_list.append(brewery_address)
                else:
                    brewery_list.append(f'They are a {brewery_type} located at {brewery_address}.')
                    addresses_list.append(None)
        
        else:
            output_list.append(rec)
            info_list.append("Unfortunately I don't have much information about this beer.")
            brewery_list.append(None)
            addresses_list.append(None)
    return output_list, info_list, brewery_list, addresses_list

def alternative_recommendations(user_beers):
    """
    Recommends beers to the user if the KNN model fails.
    Simply looks up beer style of the user's favorite beers and recommends the highest rated beers of that style.
    """
    avg_ratings_df = pd.read_csv('beer_ratings.csv')
    beer_styles = []
    rated = []
    output_list = []
    info_list = []
    brewery_list = []
    addresses_list = []
    for beer in user_beers:
        rated.append(beer)
        if beer in list(beer_df['beer']):
            beer_styles.append(beer_df[beer_df['beer']==beer]['style'].iloc[0])  # Look up their styles in the beer df

    if len(beer_styles) != 0:    # If any of the user's beers styles were retrieved

        # all beers of the same style as the user's favorites, ignoring the ones out of production and already rated ones.
        df1 = beer_df[(beer_df['style'].isin(beer_styles)) & (~beer_df['beer'].isin(rated))]
        df2 = avg_ratings_df[avg_ratings_df['beer'].isin(list(df1['beer']))].drop(columns = ['link', 'user_rating', 'user_id'])
        df3 = pd.merge(df1, df2, on='beer', how='left').drop_duplicates().sort_values(by = 'avg_rating', ascending = False)
    
        for i in range(5):
            rec = df3['beer'].iloc[i]
            style = df3['style'].iloc[i]
            status = df3['status'].iloc[i]
            brewery = df3['brewery'].iloc[i]
            if status == 'regular':
                output_list.append(rec)
                info_list.append(f'It is a {style} brewed by {brewery}.')
            else:
                output_list.append(rec)
                info_list.append(f'It is a {style}, a {status} beer brewed by {brewery}.')

            if brewery in breweries_df['brewery'].unique():
                brewery_type = breweries_df[breweries_df['brewery'] == brewery]['type'].iloc[0]
                brewery_address = breweries_df[breweries_df['brewery'] == brewery]['address'].iloc[0]
                visitable = ['Microbrewery', 'Brewpub/Brewery', 'Brewpub']
                if brewery_type in visitable:
                    brewery_list.append(f'They are a {brewery_type} and you can visit them at ')
                    addresses_list.append(brewery_address)
                else:
                    brewery_list.append(f'They are a {brewery_type} located at {brewery_address}.')
                    addresses_list.append(None)
        return output_list, info_list, brewery_list, addresses_list
    
    else:
        return None
        
def prov_beer_recommendations(predictions, users_beer_styles, prov):
    """
    Accepts a list of rating predictions produced by a surprise model and beer styles for user's favorite beers.
    Returns beers with the highest predicted rating that match user's favorite beer styles and chosen province(s). Also returns some 
    extra info about each beer (style, brewery and its address).
    """
    NUM_RECS = 5 # number of recommendations to be returned
    
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True) # sort the ratings from highest to lowest
    rec_list = []

    # dfs narrowed down to user's province(s)
    if prov == '1': # BC
        prov_df = pd.read_csv('./website/data/beer_info_bc.csv')
    elif prov == '2': # Prairies
        prov_df = pd.read_csv('./website/data/beer_info_pr.csv')
    elif prov == '3': # Ontario
        prov_df = pd.read_csv('./website/data/beer_info_on.csv')
    elif prov == '4': # Quebec
        prov_df = pd.read_csv('./website/data/beer_info_qc.csv')
    elif prov == '5': # Maritimes
        prov_df = pd.read_csv('./website/data/beer_info_ma.csv')
    else: # if somehow there is a different response, use the full df without province filters
        prov_df = pd.read_csv('./website/data/beer_info_final.csv')

    # Getting top 5 recommended beers from the list of predictions.
    for rec in recommendations:
        rec_beer = ratings_df[ratings_df['beer_id'] == rec.iid]['beer'].iloc[0] # find the beer based on its id
        current = beer_df[beer_df['beer'] == rec_beer] # look up the beer in the beer_df
        style = current['style'].iloc[0] if len(current) == 1 else None

        # If it was possible to identify user's favorite beer styles, take them into account
        if len(users_beer_styles) != 0:
            if rec_beer in list(prov_df['beer']) and style in users_beer_styles:
                rec_list.append(rec_beer)
                if len(rec_list) >= NUM_RECS:
                    break
        # Otherwise just return top recommendations regardless of style
        else:
            if rec_beer in list(prov_df['beer']):
                rec_list.append(rec_beer)
                if len(rec_list) >= NUM_RECS:
                    break

    # If beer style, status, brewery and brewery address are known, return these to the user.
    output_list = []
    info_list = []
    brewery_list = []
    addresses_list = []
    for rec in rec_list:
        current = prov_df[prov_df['beer'] == rec]
        if len(current) != 0:
            style = current['style'].iloc[0]
            status = current['status'].iloc[0]
            brewery = current['brewery'].iloc[0]
            if status == 'regular': # beer status is either regular or seasonal (e.g. summer seasonal, winter seasonal etc.)
                output_list.append(rec)
                info_list.append(f'It is a {style} brewed by {brewery}.')
            else:
                output_list.append(rec)
                info_list.append(f'It is a {style}, a {status} beer brewed by {brewery}.')

            # if the brewery is not a client brewer or a commercial brewery, provide their address
            if brewery in breweries_df['brewery'].unique():
                brewery_type = breweries_df[breweries_df['brewery'] == brewery]['type'].iloc[0]
                brewery_address = breweries_df[breweries_df['brewery'] == brewery]['address'].iloc[0]
                visitable = ['Microbrewery', 'Brewpub/Brewery', 'Brewpub']
                if brewery_type in visitable:
                    brewery_list.append(f'They are a {brewery_type} and you can visit them at ')
                    addresses_list.append(brewery_address)
                else:
                    brewery_list.append(f'They are a {brewery_type} located at {brewery_address}.')
                    addresses_list.append(None)
        
        else:
            output_list.append(rec)
            info_list.append("Unfortunately I don't have much information about this beer.")
            brewery_list.append(None)
            addresses_list.append(None)
    return output_list, info_list, brewery_list, addresses_list
