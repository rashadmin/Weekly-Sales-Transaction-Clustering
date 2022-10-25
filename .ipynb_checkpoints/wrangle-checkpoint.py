import pandas as pd
import re
def wrangle(filename):
    df = pd.read_csv(filename)
    #Dropping Normizalized Data
    col_to_drop = [i for i in df.columns if re.match(r'Normalized',i)]
    df.drop(columns=col_to_drop,inplace=True)
    #Dropping the product code
    df.drop(columns='Product_Code',inplace=True)
    return df

