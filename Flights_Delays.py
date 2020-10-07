from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from flask_table import Table, Col
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#building flask table for showing prediction results
class Results(Table):
    id = Col('Id',show=False)
    title = Col('movie_title')

app = Flask(__name__)

#Welcome Page
@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method=="POST":
        return render_template('prediction.html')
    return render_template('welcome.html')

#Results Page
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == 'POST':
        
        #reading the original dataset
        df = pd.read_csv('flights_B6.csv')
        df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(int)
        df['DAY_OF_MONTH'] = df['DAY_OF_MONTH'].astype(int)
        df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].astype(int)
        df['ARR_DELAY'] = df['ARR_DELAY'].astype(int)
        df = pd.get_dummies(df, columns = ['DEST'])
        df = pd.get_dummies(df, columns = ['ORIGIN'])
        
        X = df.drop('ARR_DELAY',axis=1)
        y = df.ARR_DELAY
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
        
        ridge_mod = Ridge()
        ridge_mod.set_params(alpha=48.24)
        ridge_mod.fit(X_train, y_train)

        #reading movie title given by user in the front-end
        Date = request.form.get('fdate')
        Origin = request.form.get('forigin')
        Dest = request.form.get('fdest')
        Hour = request.form.get('fhour')
        DateHour = str(Date) + ' ' + str(Hour)
        
        def predict_delay(departure_date_time,origin, destination):
            from datetime import datetime
            try:
                departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
            except ValueError as e:
                return 'Error parsing date/time - {}'.format(e) + ' ' + departure_date_time

            month = departure_date_time_parsed.month
            day = departure_date_time_parsed.day
            day_of_week = departure_date_time_parsed.isoweekday()
            hour = departure_date_time_parsed.hour
            hour_arrival = departure_date_time_parsed.hour

            origin = origin.upper()
            destination = destination.upper()

            input = [{'MONTH': month,
                      'CRS_DEP_TIME': hour,
                      'DAY_OF_MONTH': month,
                      'DAY_OF_WEEK': day_of_week,
                      'ORIGIN_BQN': 1 if origin == 'BQN' else 0,
                      'ORIGIN_SAV': 1 if origin == 'SAV' else 0,
                      'ORIGIN_JFK': 1 if origin == 'JFK' else 0, 
                      'ORIGIN_CHS': 1 if origin == 'CHS' else 0, 
                      'ORIGIN_DCA': 1 if origin == 'DCA' else 0, 
                      'ORIGIN_SWF': 1 if origin == 'SWF' else 0, 
                      'ORIGIN_SJU': 1 if origin == 'SJU' else 0, 
                      'ORIGIN_BOS': 1 if origin == 'BOS' else 0, 
                      'ORIGIN_RSW': 1 if origin == 'RSW' else 0,
                      'ORIGIN_ORD': 1 if origin == 'ORD' else 0, 
                      'ORIGIN_LGB': 1 if origin == 'LGB' else 0, 
                      'ORIGIN_LAS': 1 if origin == 'LAS' else 0, 
                      'ORIGIN_MCO': 1 if origin == 'MCO' else 0, 
                      'ORIGIN_RDU': 1 if origin == 'RDU' else 0, 
                      'ORIGIN_TPA': 1 if origin == 'TPA' else 0, 
                      'ORIGIN_BDL': 1 if origin == 'BDL' else 0, 
                      'ORIGIN_SFO': 1 if origin == 'SFO' else 0, 
                      'ORIGIN_PBI': 1 if origin == 'PBI' else 0,
                      'ORIGIN_EWR': 1 if origin == 'EWR' else 0, 
                      'ORIGIN_FLL': 1 if origin == 'FLL' else 0, 
                      'ORIGIN_LGA': 1 if origin == 'LGA' else 0, 
                      'ORIGIN_PHL': 1 if origin == 'PHL' else 0, 
                      'ORIGIN_DEN': 1 if origin == 'DEN' else 0, 
                      'ORIGIN_SEA': 1 if origin == 'SEA' else 0, 
                      'ORIGIN_JAX': 1 if origin == 'JAX' else 0, 
                      'ORIGIN_BWI': 1 if origin == 'BWI' else 0, 
                      'ORIGIN_HOU': 1 if origin == 'HOU' else 0,
                      'ORIGIN_STT': 1 if origin == 'STT' else 0, 
                      'ORIGIN_AUS': 1 if origin == 'AUS' else 0, 
                      'ORIGIN_PVD': 1 if origin == 'PVD' else 0, 
                      'ORIGIN_RIC': 1 if origin == 'RIC' else 0, 
                      'ORIGIN_DFW': 1 if origin == 'DFW' else 0, 
                      'ORIGIN_CLT': 1 if origin == 'CLT' else 0, 
                      'ORIGIN_LAX': 1 if origin == 'LAX' else 0, 
                      'ORIGIN_DTW': 1 if origin == 'DTW' else 0, 
                      'ORIGIN_HPN': 1 if origin == 'HPN' else 0,
                      'ORIGIN_PDX': 1 if origin == 'PDX' else 0, 
                      'ORIGIN_BUF': 1 if origin == 'BUF' else 0, 
                      'ORIGIN_PIT': 1 if origin == 'PIT' else 0, 
                      'ORIGIN_SAN': 1 if origin == 'SAN' else 0, 
                      'ORIGIN_PWM': 1 if origin == 'PWM' else 0, 
                      'ORIGIN_SLC': 1 if origin == 'SLC' else 0, 
                      'ORIGIN_BTV': 1 if origin == 'BTV' else 0, 
                      'ORIGIN_SMF': 1 if origin == 'SMF' else 0, 
                      'ORIGIN_STX': 1 if origin == 'STX' else 0,
                      'ORIGIN_OAK': 1 if origin == 'OAK' else 0, 
                      'ORIGIN_IAD': 1 if origin == 'IAD' else 0, 
                      'ORIGIN_BUR': 1 if origin == 'BUR' else 0, 
                      'ORIGIN_SYR': 1 if origin == 'SYR' else 0, 
                      'ORIGIN_DAB': 1 if origin == 'DAB' else 0, 
                      'ORIGIN_MSY': 1 if origin == 'MSY' else 0, 
                      'ORIGIN_SRQ': 1 if origin == 'SRQ' else 0, 
                      'ORIGIN_CLE': 1 if origin == 'CLE' else 0, 
                      'ORIGIN_ROC': 1 if origin == 'ROC' else 0,
                      'ORIGIN_PHX': 1 if origin == 'PHX' else 0,
                      'ORIGIN_PSE': 1 if origin == 'PSE' else 0, 
                      'ORIGIN_ORH': 1 if origin == 'ORH' else 0, 
                      'ORIGIN_RNO': 1 if origin == 'RNO' else 0, 
                      'ORIGIN_ALB': 1 if origin == 'ALB' else 0, 
                      'ORIGIN_ABQ': 1 if origin == 'ABQ' else 0, 
                      'ORIGIN_PSP': 1 if origin == 'PSP' else 0, 
                      'ORIGIN_SJC': 1 if origin == 'SJC' else 0,
                      'ORIGIN_BNA': 1 if origin == 'BNA' else 0,
                      'ORIGIN_ACK': 1 if origin == 'ACK' else 0, 
                      'ORIGIN_ANC': 1 if origin == 'ANC' else 0, 
                      'ORIGIN_MVY': 1 if origin == 'MVY' else 0, 
                      'ORIGIN_HYA': 1 if origin == 'HYA' else 0,
                      'DEST_BQN': 1 if destination == 'BQN' else 0,
                      'DEST_SAV': 1 if destination == 'SAV' else 0,
                      'DEST_JFK': 1 if destination == 'JFK' else 0, 
                      'DEST_CHS': 1 if destination == 'CHS' else 0, 
                      'DEST_DCA': 1 if destination == 'DCA' else 0, 
                      'DEST_SWF': 1 if destination == 'SWF' else 0, 
                      'DEST_SJU': 1 if destination == 'SJU' else 0, 
                      'DEST_BOS': 1 if destination == 'BOS' else 0, 
                      'DEST_RSW': 1 if destination == 'RSW' else 0,
                      'DEST_ORD': 1 if destination == 'ORD' else 0, 
                      'DEST_LGB': 1 if destination == 'LGB' else 0, 
                      'DEST_LAS': 1 if destination == 'LAS' else 0, 
                      'DEST_MCO': 1 if destination == 'MCO' else 0, 
                      'DEST_RDU': 1 if destination == 'RDU' else 0, 
                      'DEST_TPA': 1 if destination == 'TPA' else 0, 
                      'DEST_BDL': 1 if destination == 'BDL' else 0, 
                      'DEST_SFO': 1 if destination == 'SFO' else 0, 
                      'DEST_PBI': 1 if destination == 'PBI' else 0,
                      'DEST_EWR': 1 if destination == 'EWR' else 0, 
                      'DEST_FLL': 1 if destination == 'FLL' else 0, 
                      'DEST_LGA': 1 if destination == 'LGA' else 0, 
                      'DEST_PHL': 1 if destination == 'PHL' else 0, 
                      'DEST_DEN': 1 if destination == 'DEN' else 0, 
                      'DEST_SEA': 1 if destination == 'SEA' else 0, 
                      'DEST_JAX': 1 if destination == 'JAX' else 0, 
                      'DEST_BWI': 1 if destination == 'BWI' else 0, 
                      'DEST_HOU': 1 if destination == 'HOU' else 0,
                      'DEST_STT': 1 if destination == 'STT' else 0, 
                      'DEST_AUS': 1 if destination == 'AUS' else 0, 
                      'DEST_PVD': 1 if destination == 'PVD' else 0, 
                      'DEST_RIC': 1 if destination == 'RIC' else 0, 
                      'DEST_DFW': 1 if destination == 'DFW' else 0, 
                      'DEST_CLT': 1 if destination == 'CLT' else 0, 
                      'DEST_LAX': 1 if destination == 'LAX' else 0, 
                      'DEST_DTW': 1 if destination == 'DTW' else 0, 
                      'DEST_HPN': 1 if destination == 'HPN' else 0,
                      'DEST_PDX': 1 if destination == 'PDX' else 0, 
                      'DEST_BUF': 1 if destination == 'BUF' else 0, 
                      'DEST_PIT': 1 if destination == 'PIT' else 0, 
                      'DEST_SAN': 1 if destination == 'SAN' else 0, 
                      'DEST_PWM': 1 if destination == 'PWM' else 0, 
                      'DEST_SLC': 1 if destination == 'SLC' else 0, 
                      'DEST_BTV': 1 if destination == 'BTV' else 0, 
                      'DEST_SMF': 1 if destination == 'SMF' else 0, 
                      'DEST_STX': 1 if destination == 'STX' else 0,
                      'DEST_OAK': 1 if destination == 'OAK' else 0, 
                      'DEST_IAD': 1 if destination == 'IAD' else 0, 
                      'DEST_BUR': 1 if destination == 'BUR' else 0, 
                      'DEST_SYR': 1 if destination == 'SYR' else 0, 
                      'DEST_DAB': 1 if destination == 'DAB' else 0, 
                      'DEST_MSY': 1 if destination == 'MSY' else 0, 
                      'DEST_SRQ': 1 if destination == 'SRQ' else 0, 
                      'DEST_CLE': 1 if destination == 'CLE' else 0, 
                      'DEST_ROC': 1 if destination == 'ROC' else 0,
                      'DEST_PHX': 1 if destination == 'PHX' else 0,
                      'DEST_PSE': 1 if destination == 'PSE' else 0, 
                      'DEST_ORH': 1 if destination == 'ORH' else 0, 
                      'DEST_RNO': 1 if destination == 'RNO' else 0, 
                      'DEST_ALB': 1 if destination == 'ALB' else 0, 
                      'DEST_ABQ': 1 if destination == 'ABQ' else 0, 
                      'DEST_PSP': 1 if destination == 'PSP' else 0, 
                      'DEST_SJC': 1 if destination == 'SJC' else 0,
                      'DEST_BNA': 1 if destination == 'BNA' else 0,
                      'DEST_ACK': 1 if destination == 'ACK' else 0, 
                      'DEST_ANC': 1 if destination == 'ANC' else 0, 
                      'DEST_MVY': 1 if destination == 'MVY' else 0, 
                      'DEST_HYA': 1 if destination == 'HYA' else 0}]

             # Now predict this with the model 

            pred_delay = ridge_mod.predict(pd.DataFrame(input))
            return int(pred_delay[0])

        try:
            output = predict_delay(DateHour,Origin,Dest)
            return render_template('prediction.html', output=output)
        except ValueError as e:
            return render_template('welcome.html', error=e)

if __name__ == '__main__':
   app.run(debug = True)
