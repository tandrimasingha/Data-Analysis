#importing the required libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go


#creation of the Flask Application named as "app"
app = Flask(__name__)

#loading the pickle files of models which is used in read binary mode
model = pickle.load(open('random_regressor.pkl', 'rb'))

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

#home page - routing to the home page is done
@app.route('/')
def home():
    #renders the home page template 
    return render_template('index.html')


#routing to the car price prediction page
@app.route('/i')
def i():
    return render_template('i.html')

#routing to the Car Sales Analysis in Ukraine page
@app.route('/z',methods=['GET'])
#portion for data visualization and analysis for Car Sales Analysis in Ukraine
def visualize1():
     #reading the dataset
    carsales_df = pd.read_csv('car_ad.csv',encoding='ISO-8859-1')
    df = pd.DataFrame(carsales_df.car.value_counts())
    #Histogram plot of car brand along with sales
    fig = px.histogram(carsales_df,
                   x='car',
                   color='car',
                   marginal='box',
                   title='Car Brand along with their Sales')
    fig.update_layout(bargap=0.1)
    #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    carsales_year_df = pd.DataFrame(carsales_df.groupby('year').car.value_counts())
    carsales_year_df.rename(columns={'car':'sales'}, inplace=True)
    carsales_year_df.reset_index(inplace=True)
    topCarBrandSales = carsales_year_df[carsales_year_df.car.isin(df.head(5).index)]
    fig2= px.line(topCarBrandSales, x="year", y="sales", color='car') # line plot for top car brand sales
     #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    recentCarSalesTopBrands = carsales_year_df[carsales_year_df.car.isin(df.head(5).index) &  (carsales_year_df.year >= 2010)]
    
    fig3=px.line(recentCarSalesTopBrands, x='year', y='sales', color='car')
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
     
    # function to get sales by year
    def get_SalesByYear(year):
        return carsales_year_df[carsales_year_df.year == year]
    # function to get sales rank by year
    def get_CarSalesRankByYear(r):
        result = get_SalesByYear(r.year).sales.unique()
        i, = np.where(result == r.sales)
        return i[0]+1

    # preprocessing the datset to get the analysis perfectly.
    carsales_year_df['year_rank'] = carsales_year_df.apply(get_CarSalesRankByYear, axis=1)
    recenttopcars = carsales_year_df[(carsales_year_df.year_rank <=5) &  (carsales_year_df.year >= 2009)].car.unique()
    topcarbrands = carsales_year_df[(carsales_year_df.year_rank <=5) &  (carsales_year_df.year >= 1980)].car.unique()
    recentCarSalesRanks = carsales_year_df[carsales_year_df.car.isin(recenttopcars) & (carsales_year_df.year >= 2009)]
    recentCarSalesRanks.pivot_table(index=['year'], columns={'car'}, values='sales')
    recentCarSalesRanks.pivot_table(index=['year'], columns={'car'}, values='year_rank')
    fig4=px.scatter(recentCarSalesRanks, x='year', y='sales', color='car')
    #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig5=px.scatter(recentCarSalesRanks, x='year', y='year_rank', color='car')
    #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    carsales_df[carsales_df.price.isin(carsales_df.price.nlargest(10))].sort_values('price', ascending=False)
    cars_with_max_price_df = pd.DataFrame(carsales_df.groupby('car').price.max())
    cars_with_max_price_df.reset_index(inplace=True)
    cars_with_max_price_df.rename(columns={'price':'max_price'}, inplace=True)
    def get_totalcarsales(r):
        return carsales_df[(carsales_df.car == r.car) & (carsales_df.price == r.max_price)].car.count()

    def get_latestyearofsale(r):
        return carsales_df[(carsales_df.car == r.car) & (carsales_df.price == r.max_price)].year.max()
    cars_with_max_price_df['total_sales'] = cars_with_max_price_df.apply(get_totalcarsales, axis=1)
    cars_with_max_price_df['recently_sold_on'] = cars_with_max_price_df.apply(get_latestyearofsale, axis=1)
    
    fig6=px.scatter(cars_with_max_price_df.sort_values('max_price', ascending=False).head(10), x='recently_sold_on', y='max_price', color='car')
    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    # classifing the class column in the dataset
    def classify_class(r):
        if r.price <= 10000:
            return "Economy"
        elif (r.price > 10000) & (r.price <= 30000):
            return "Luxury"
        else:
            return "Premium"
# Set new column called 'class' for defining the sement

    carsales_df["class"] = carsales_df.apply(classify_class, axis=1)
    fig7 = px.histogram(carsales_df,
                   x='class',
                   color='class',
                   marginal='box',
                   title='Car Brand along with their Sales')
    fig.update_layout(bargap=0.1)
    graphJSON7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)
    x=carsales_df.car.value_counts()
    
    fig8 = px.pie(carsales_df, names='class', title='Population of European continent')
    graphJSON8 = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)
    
    pvt_classyear_sales = carsales_df.pivot_table(index=['year','class'],  values="car", aggfunc='count')
    pvt_classyear_sales.reset_index(inplace=True)
    pvt_classyear_sales.rename(columns={"car":"car_sales"}, inplace=True)
    '''
    Function that returns revenue of a perticular segment in a specific year
    '''
    def get_revenuebyclass(r):
        return carsales_df[(carsales_df['class'] == r['class']) & (carsales_df.year == r.year)].price.sum()
        
    pvt_classyear_sales['revenue'] = pvt_classyear_sales.apply(get_revenuebyclass, axis=1)
    def get_detailsbyclass(cls):
        return pvt_classyear_sales[pvt_classyear_sales['class'] == cls]

    def get_detailsbyclass(cls, year):
        return pvt_classyear_sales[(pvt_classyear_sales['class'] == cls) & (pvt_classyear_sales.year >= year)]
    
    fig9=px.line(pvt_classyear_sales[pvt_classyear_sales.year >= 1990], x='year', y='car_sales', color='class')
    graphJSON9 = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig10=px.line(pvt_classyear_sales[pvt_classyear_sales.year >= 1990], x='year', y='revenue', color='class')
    graphJSON10 = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)
    
    carsalesbybody_df = carsales_df.pivot_table(index=['body'],  values="car", aggfunc='count')
    carsalesbybody_df.reset_index(inplace=True)
    carsalesbybody_df.rename(columns={'car':'sales'}, inplace=True)
    recent_carsalesbybody_df = carsales_df[carsales_df.year > 2010].pivot_table(index=['body'],  values="car", aggfunc='count')
    recent_carsalesbybody_df.reset_index(inplace=True)
    recent_carsalesbybody_df.rename(columns={'car':'sales'}, inplace=True)
    pvt_bodyyear_sales = carsales_df.pivot_table(index=['year','body'],  values="car", aggfunc='count')
    pvt_bodyyear_sales.rename(columns={"car":"car_sales"}, inplace=True)
    pvt_bodyyear_sales.reset_index(inplace=True)

    def get_sales_by_body(body):
        return pvt_bodyyear_sales[pvt_bodyyear_sales['body'] == body]

    # crossover, sedan, van, vagon, hatch, other
    
    fig11=px.bar(pvt_bodyyear_sales, x='body', y='car_sales',color='year')
    graphJSON11 = json.dumps(fig11, cls=plotly.utils.PlotlyJSONEncoder)
    fig12=px.bar(carsalesbybody_df, x='body', y='sales')
    graphJSON12 = json.dumps(fig12, cls=plotly.utils.PlotlyJSONEncoder)
    fig13=px.bar(recent_carsalesbybody_df, x='body', y='sales')
    graphJSON13 = json.dumps(fig13, cls=plotly.utils.PlotlyJSONEncoder)
    fig14=px.line(pvt_bodyyear_sales[pvt_bodyyear_sales.year >= 1990], x='year', y='car_sales', color='body')
    graphJSON14 = json.dumps(fig14, cls=plotly.utils.PlotlyJSONEncoder)
    
    besteconomycars = carsales_df.pivot_table(index=['class','car'],  values="mileage", aggfunc='max')
    besteconomycars.reset_index(inplace=True)
    # Return upper whisker for mileage by class
    def get_best_carandmileage_byclass(cls):
        clsmileage = besteconomycars[besteconomycars['class'] == cls]
        maxval = clsmileage.mileage.max()
        thirdQurtile = clsmileage.describe().iloc[6].mileage   
        firstQurtile = clsmileage.describe().iloc[4].mileage     
        iqr = thirdQurtile - firstQurtile
        upperwhisker = min(maxval, thirdQurtile + (1.5 * iqr))
        return upperwhisker

    # Get closest upper whisker milage results
    def get_closest(cls, val):
        cls_df = carsales_df[carsales_df['class'] == cls]
        return cls_df.iloc[(cls_df['mileage']-val).abs().argsort()[:2]]
    
    fig15=px.box(besteconomycars, x='class', y='mileage', points='all', boxmode="overlay")
    graphJSON15 = json.dumps(fig15, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig16=px.histogram(carsales_df,x="engType", y="price", color= "body",title='Average price of vehicles by engine type and drive')
    graphJSON16 = json.dumps(fig16, cls=plotly.utils.PlotlyJSONEncoder)
    fig17=px.histogram(carsales_df,x='drive', title='Overall car sales by drive')
    fig18=px.histogram(carsales_df[carsales_df.year > 2010],x='drive', title='Latest 5 year car sales by drive')
    graphJSON17 = json.dumps(fig17, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON18 = json.dumps(fig18, cls=plotly.utils.PlotlyJSONEncoder)
    
    salesByEngTypeBodyClass_df = carsales_df[carsales_df.year.isin(pd.DataFrame(carsales_df.year.unique())[0].nlargest(10))]
    salesByYearBodyClass = salesByEngTypeBodyClass_df.pivot_table(index=['year','engType'], values='price', aggfunc='mean')
    salesByYearBodyClass.reset_index(inplace=True)

    def get_priceByYearEngType(r):
        return carsales_df[(carsales_df.year == r.year) & (carsales_df.engType == r.engType)].car.count()
    salesByYearBodyClass['sales'] = salesByYearBodyClass.apply(get_priceByYearEngType, axis=1)
    
    fig19=px.scatter_matrix(carsales_df, dimensions=["car", "price", "body"])
    graphJSON19 = json.dumps(fig19, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig20=px.scatter(salesByYearBodyClass, x='sales', y='price', color='engType')
    graphJSON20 = json.dumps(fig20, cls=plotly.utils.PlotlyJSONEncoder)
    
    carsalesold_df = carsales_df[carsales_df.year < 2012].groupby('year')['engType'].value_counts()
    carsalesold_pv = pd.DataFrame(carsalesold_df)
    carsalesold_pv.rename(columns={'engType':'sales'}, inplace=True)
    carsalesold_pv.reset_index(inplace=True)
    fig21=px.pie(carsalesold_pv,values='sales', names='engType')
    graphJSON21 = json.dumps(fig21, cls=plotly.utils.PlotlyJSONEncoder)
    
    classsalesByEngType_5year = carsales_df[carsales_df.year.isin(pd.DataFrame(carsales_df.year.unique())[0].nlargest(10))].groupby('year')['engType'].value_counts()
    carsalesEngType = pd.DataFrame(classsalesByEngType_5year)
    carsalesEngType.rename(columns={'engType':'sales'}, inplace=True)
    carsalesEngType.reset_index(inplace=True)
    
    fig22=px.pie(carsalesEngType,values='sales', names='engType')
    graphJSON22 = json.dumps(fig22, cls=plotly.utils.PlotlyJSONEncoder)
    
    
   
    ## this line tells Flask to use an HTML template called visual1.html and pass to it the JSON code
    return render_template('visual1.html', graphJSON=graphJSON,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,
                           graphJSON5=graphJSON5,graphJSON6=graphJSON6,graphJSON7=graphJSON7,graphJSON8=graphJSON8,graphJSON9=graphJSON9,
                           graphJSON10=graphJSON10,graphJSON11=graphJSON11,graphJSON12=graphJSON12,graphJSON13=graphJSON13,
                           graphJSON14=graphJSON14,graphJSON15=graphJSON15,graphJSON16=graphJSON16,graphJSON17=graphJSON17,graphJSON18=graphJSON18,
                           graphJSON19=graphJSON19,graphJSON20=graphJSON20,graphJSON21=graphJSON21,graphJSON22=graphJSON22)
    


@app.route('/j',methods=['GET'])
def visualize2():
    carSales = pd.read_csv('Car_sales.csv',encoding='ISO-8859-1')
    fig = px.scatter(carSales,
                 x='Power_perf_factor',
                 y='Price_in_thousands',
                 color='Model',
                
                 hover_data=['Price_in_thousands'],
                 title='Price vs. Model')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    carSales.Power_perf_factor =carSales.Power_perf_factor.replace(np.nan, 0, regex=True)
    carSales.__year_resale_value =carSales.__year_resale_value.replace(np.nan, 0, regex=True)
    carSales.Fuel_capacity=carSales.Fuel_capacity.replace(np.nan,0,regex=True)
    
    
        
    fig2=px.histogram(carSales,x='Manufacturer',color='Manufacturer',title='Analysis manufacturers with most models manufactured')

    
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    trace=go.Histogram(
    x=carSales.Power_perf_factor)
    
    layout = go.Layout(
        title={
            'text':' Histogram of Power_perf_factor',
            'y':0.9,
            'x':0.5,
            'xanchor': 'left',
            'yanchor': 'top' 
            },          
        bargap=0.2,
        xaxis=dict(title='Power_perf_factor'),
        yaxis=dict( title='Count'),
    )

    fig3 = go.Figure(data=trace, layout=layout)
    fig3.update_traces(opacity=0.75)
    
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig4 = px.histogram(carSales,
                   x='Power_perf_factor',
                   marginal='box',
                   color='Model',
                   title='Distribution of Power Perform Factor',
                   )
    fig4.update_layout(bargap=0.1)
    
    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig5=px.histogram(carSales,
                 x="__year_resale_value",
                 marginal="box",
                 title='Distribution of Year Resale Value',
    )
    fig5.update_layout(bargap=0.1)
    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig6=px.bar(carSales,
                 y="Sales_in_thousands",
                 x="Model",
                 color='Model',
                 title='Distribution of Sales in Thousansds',
    )
    fig6.update_layout(bargap=0.1)
    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig7=px.histogram(carSales,
                 x="Vehicle_type",
                
                 title='Distribution of Vehicle Type',
                
    )
    fig7.update_layout(bargap=0.1)
    
    graphJSON7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig8 = px.bar(carSales,
                   y='Price_in_thousands',
                   x='Manufacturer',
                   title='Distribution of Price in thousand')
    fig8.update_layout(bargap=0.1)
    
    graphJSON8 = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig9 = px.histogram(carSales,
                   x='Price_in_thousands',
                   marginal='box',
                   title='Distribution of Price in thousand')
    fig9.update_layout(bargap=0.1)

    graphJSON9 = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    carBrand=list(carSales['Manufacturer'].unique())
    carType=list(carSales['Vehicle_type'].unique())
    salesThousand_ratio=[]
    yearResaleValue_ratio=[]
    priceThousand_ratio=[]
    for i in carBrand:
        x=carSales[carSales['Manufacturer']==i]
        salesThousand_rate=sum(x.Sales_in_thousands)/len(x)
        salesThousand_ratio.append(salesThousand_rate)
    datasalesTousands=pd.DataFrame({"car_brand":carBrand,"salesThousand_ratio":salesThousand_ratio})
    new_index=(datasalesTousands['salesThousand_ratio'].sort_values(ascending=False)).index.values
    shorted_salesThousandsData=datasalesTousands.reindex(new_index)

    trace=go.Bar(
    x=shorted_salesThousandsData.car_brand,
    y=shorted_salesThousandsData.salesThousand_ratio,
    type='bar',
    marker=dict(color=shorted_salesThousandsData.salesThousand_ratio,line=dict(color='rgb(0,0,0)',width=1.5)),
    text=shorted_salesThousandsData.car_brand
    )
    data=[trace]
    layout=dict(title="Car brand sales thousands ratio",
            xaxis=dict(title="car_brand"),
            yaxis=dict(title="salesThousand_ratio"),
            barmode='relative')#If we don't, they stand side by side.



    # fig = dict(data = data, layout = layout)
    fig10=go.Figure(data=data,layout=layout)
    # iplot(fig)
    fig10.update_layout(barmode='relative',
        title={
            
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top','font_color':'rgba(128, 0, 0,0.5)'})
    
    graphJSON10 = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig11 = px.scatter(carSales,
                 x='Sales_in_thousands',
                 y='Price_in_thousands',
                 title='Sales_in_thousands vs. Price in Thousands',
                  color='Model')
    fig11.update_traces(marker_size=5)
    graphJSON11 = json.dumps(fig11, cls=plotly.utils.PlotlyJSONEncoder)
   
    yearResaleValue_ratioType=[]
    for i in carType:
        x=carSales[carSales['Vehicle_type']==i]
        x.__year_resale_value=x.__year_resale_value.replace(np.nan,0)
        yearResaleValue_rate=sum(x.__year_resale_value)/len(x)
        yearResaleValue_ratioType.append(yearResaleValue_rate)
    datayearResaleValuecartype=pd.DataFrame({"car_type":carType,"yearResaleValue_ratio":yearResaleValue_ratioType})
    
    fig12 = px.bar(datayearResaleValuecartype,
                 x='car_type',
                 y='yearResaleValue_ratio',
                 title="Car types year resale ratio",
                 color="car_type")
    fig12.update_traces()

    graphJSON12 = json.dumps(fig12, cls=plotly.utils.PlotlyJSONEncoder)
    
    for i in carBrand:
        x=carSales[carSales['Manufacturer']==i]
        yearResaleValue_rate=sum(x.__year_resale_value)/len(x)
        yearResaleValue_ratio.append(yearResaleValue_rate)
    datayearResaleValue=pd.DataFrame({"car_brand":carBrand,"yearResaleValue_ratio":yearResaleValue_ratio})
    new_index=(datayearResaleValue['yearResaleValue_ratio'].sort_values(ascending=False)).index.values
    shorted_yearResaleValueData=datayearResaleValue.reindex(new_index)
    
    fig13 = px.bar(shorted_yearResaleValueData,
                 x='car_brand',
                 y='yearResaleValue_ratio',
                 title='Car brand year resale ratio',
                 color="car_brand")
    fig13.update_traces()
    graphJSON13 = json.dumps(fig13, cls=plotly.utils.PlotlyJSONEncoder)
    
    for i in carBrand:
        x=carSales[carSales['Manufacturer']==i]
        priceThousand_rate=sum(x.Price_in_thousands)/len(x)
        priceThousand_ratio.append(priceThousand_rate)
    datapriceThousand=pd.DataFrame({"car_brand":carBrand,"priceThousand_ratio":priceThousand_ratio})
    new_index=(datapriceThousand['priceThousand_ratio'].sort_values(ascending=False)).index.values
    shorted_priceThousandData=datapriceThousand.reindex(new_index)
    fig14 = px.bar(shorted_priceThousandData,
                 x='car_brand',
                 y='priceThousand_ratio',
                 title='Car brand price thousand ratio',
                 color="car_brand")
    fig14.update_traces()
    graphJSON14 = json.dumps(fig14, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig15 = px.scatter(shorted_yearResaleValueData,
                 x='car_brand',
                 y='yearResaleValue_ratio',
                 title='Car brand year resale ratio',
                 color="car_brand",
                 size='yearResaleValue_ratio')
    fig15.update_traces()
    graphJSON15 = json.dumps(fig15, cls=plotly.utils.PlotlyJSONEncoder)
    
    data = [
    {
        'y': carSales.Engine_size,
        'x': carSales.Horsepower,
        'mode': 'markers',
        'marker': {
            'color':carSales.Wheelbase,
            'size':(carSales.Power_perf_factor/6),
            'showscale': True,
            'sizemin':4,
            'sizemode':'diameter',
            'symbol':'diamond-open'
           
        },
        "text" :  carSales.Model,
        
        }
    ]
    layout={
    'xaxis':{'title':'Horsepower'},
    'yaxis':{'title':'Engine_size'},
    
   
    };

    fig16=go.Figure(data=data,layout=layout)

    fig16.update_layout(
        title={
            'text': "Vehicle engin size  vs horsepower  with .Power_perf_factor(size) and .Wheelbase (color) ",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    
    graphJSON16 = json.dumps(fig16, cls=plotly.utils.PlotlyJSONEncoder)
    
    carSales.Engine_size=carSales.Engine_size.replace(np.nan,0,regex=True)

    fig17=px.scatter_3d(carSales,
                    x='Fuel_capacity',
                    y='Fuel_efficiency',
                    z='Horsepower',
                    color='Model',
                    size='Engine_size',
                    
                    
    )

    fig17.update_layout(scene = dict(
                        
    #                     xaxis_title='X AXIS TITLE',
    #                     yaxis_title='Y AXIS TITLE',
    #                     zaxis_title='Z AXIS TITLE'),
    #                     width=700,
    #                     margin=dict(r=20, b=10, l=10, t=10),
                        xaxis = dict(
                            title="Sales_in_thousands",
                            backgroundcolor="rgb(250, 210, 230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",),
                        yaxis = dict(
                            title="year_resale_value",
                            backgroundcolor="rgb(230, 250,330)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),
                        zaxis = dict(
                            title="Price_in_thousands",
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",)
                        
                    
                    ),
                    title={
                            'text':'Vehicle Sales_in_thousands, year_resale_value, Price_in_thousands rates size(Fuel CApacity)',
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top' })
    graphJSON17 = json.dumps(fig17, cls=plotly.utils.PlotlyJSONEncoder)
    
    # this line tells Flask to use an HTML template called visual2.html and pass to it the JSON code
    return render_template('visual2.html', graphJSON=graphJSON, graphJSON2=graphJSON2, graphJSON3=graphJSON3, graphJSON4=graphJSON4,graphJSON5=graphJSON5,
                           graphJSON6=graphJSON6,graphJSON7=graphJSON7,graphJSON8=graphJSON8,graphJSON9=graphJSON9,graphJSON10=graphJSON10,
                           graphJSON11=graphJSON11,graphJSON12=graphJSON12,graphJSON13=graphJSON13,graphJSON14=graphJSON14,graphJSON15=graphJSON15,graphJSON16=graphJSON16,graphJSON17=graphJSON17)



# car price prediction
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        # input for the form 
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        Year=2021-Year
        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0	
        Transmission_Manual=request.form['Transmission_Manual']
        if(Transmission_Manual=='Manual'):
            Transmission_Manual=1
        else:
            Transmission_Manual=0
         
        #predict the output on basis of the features fed to the model
        prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]])
        output=round(prediction[0],2)
         #on basis of prediction displaying the desired output
        if output<0:
            return render_template('i.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('i.html',prediction_text="You Can Sell The Car at {} lakh".format(output))
    else:
        return render_template('i.html')

#debug is set to True in development environment and set to False in production environment
if __name__=="__main__":
    app.run(debug=True)
