import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def error_reduction(a, b):
    return abs(a-b)*100/b

dep_lang_dic = { 'UD_Ligurian-GLT': 'lig',
    'UD_Bhojpuri-BHTB': 'bho',
    'UD_Kiche-IU': 'kic',
    'UD_Guajajara-TuDeT': 'gua',
    'UD_Skolt_Sami-Giellagas': 'sam',
    'UD_Armenian-ArmTDP': 'arm',
    'UD_Welsh-CCG': 'cym',
    'UD_Vietnamese-VTB': 'vie',
    'UD_Basque-BDT': 'eus',
    'UD_Bulgarian-BTB': 'bul',
    'UD_Turkish-BOUN': 'tur', 
    'UD_Ancient_Greek-Perseus': 'agk', 
    'UD_Chinese-GSDSimp': 'zho',
    'UD_Classical_Chinese-Kyoto': 'lzh',
    'UD_Naija-NSC': 'nai',
    'UD_Maltese-MUDT': 'mal',
    'UD_Gothic-PROIEL': 'got',
    'UD_Wolof-WTB': 'wol',
    'UD_Old_East_Slavic-TOROT': 'oes',}

dep_lang_order = ['sam', 'gua', 'lig', 'bho',
'kic', 'mal', 'wol', 'cym', 'arm', 'vie' ,'zho', 
'got', 'eus', 'nai',
'tur', 'bul', 'agk', 'oes', 'lzh']

dep_order_dict = {lang: i for i, lang in enumerate(dep_lang_order)}
const_lang_oder = ['swedish', 'hebrew', 'polish', 'basque', 'hungarian', 'french', 'korean', 'english', 'german', 'chinese']
const_order_dict = {lang: i for i, lang in enumerate(const_lang_oder)}

lm_dic = {
    'bert-base-multilingual-cased': 'mBERT',
    'xlm-roberta-base': 'xlm-roberta',
    'google/canine-s': 'canine-s',
    'google/canine-c': 'canine-c',
}

lm_order_dic = {'mBERT': 1, 'xlm-roberta': 2, 'canine-c': 3, 'canine-s':4}

# dependencies
df = pd.read_csv('dep_scores.csv')
df['Treebank'] = df['Treebank'].map(lambda x: dep_lang_dic[x])
df['Language Model'] = df['Language Model'].map(lambda x: lm_dic[x])
df = df[df['Finetuned'] == 'not_finetuned']
# eliminate repeated rows
df = df.drop_duplicates()
encodings = ['2-planar-brackets-greedy','rel-pos', 'arc-hybrid', 'relative']
# Create different dataframes for each plot; for an encoding/language model combination, we compare the LAS score of the finetuned/pretrained, non-finetuned/pretrained and finetuned/non-pretrained models
dataframes = {}
#palette = sns.color_palette(['#E0524D', '#DCE063', '#3694E0'])
for encoding in encodings:
    fig = plt.figure()
    df_temp = df[(df['Encoding'] == encoding)] # & (df['Language Model'] == language_model)]
    key = encoding
    dataframes[key] = df_temp
    df_temp = df_temp.drop(columns=['Unnamed: 10'])
    # Groupby treebank and language model and take the differences of the LAS scores
    df_temp = df_temp.sort_values(by=['Treebank', 'Language Model', 'Pretrained'], ascending=[True, True, False])
    df_temp['Error'] = df_temp['LAS'].map(lambda x: 100-x)
    #print(df_temp[(df_temp["Treebank"] == "bulgarian") & (df_temp["Language Model"] == "mBERT")])
    df_temp["Error reduction"] = df_temp.groupby(['Treebank', 'Language Model'])['Error'].diff()
    df_temp = df_temp.dropna()
    df_temp["Relative error reduction"] = (df_temp["Error reduction"] / df_temp["Error"]) * 100
    #print(df_temp[(df_temp["Treebank"] == "bulgarian") & (df_temp["Language Model"] == "mBERT")])
    # drop nan
    
    sns.set(font_scale=0.5, style='whitegrid')
    sns.set_context('paper', font_scale=1.25)
    df_temp['order'] = df_temp['Treebank'].map(lambda x: dep_order_dict[x])
    df_temp['order2'] = df_temp['Language Model'].map(lambda x: lm_order_dic[x])
    df_temp = df_temp.sort_values(by=['order', 'order2'], ascending=[True, True])
    plot = sns.barplot(x='Treebank', y='Relative error reduction', hue='Language Model', data=df_temp, palette="Set2", width=0.8).get_figure()
    #plot = sns.lineplot(x='Treebank', y='LAS', hue='Language Model', style="Pretrained", data=df_temp, palette="Set2", markers=True).get_figure()
    #sns.barplot(x='Treebank', y='LAS', hue='Language Model', data=df_temp, palette="Set2").get_figure()
    plt.legend(loc='best',ncol=1)
    plt.gca().xaxis.grid(False)
    
    #plt.title('LAS difference for ' + encoding)
    plt.xticks(rotation=45)
    plt.ylim(-50,54)
    sns.despine(right=True)
    
    key = key.replace('/', '_')
    plot.savefig('plots/' + key + '.png', bbox_inches='tight', dpi=300)

# Constituency
df_temp = pd.read_csv('const_scores.csv')
df_temp = df_temp[df_temp['finetuned'] == 'not_finetuned']
df_temp = df_temp.drop_duplicates()
df_temp['Treebank'] = df_temp['language']
df_temp['Language Model'] = df_temp['lm'].map(lambda x: lm_dic[x])

df_temp['Pretrained'] = df_temp['pretrained']
df_temp['F-Score'] = df_temp['BracketingFMeasure']
# drop old columns
df_temp = df_temp.drop(columns=['Unnamed: 16', 'language', 'lm', 'pretrained', 'BracketingFMeasure'])

df_temp = df_temp.sort_values(by=['Treebank', 'Language Model', 'Pretrained'], ascending=[True, True, False])
df_temp['Error'] = df_temp['F-Score'].map(lambda x: 100-x)
#print(df_temp[(df_temp["Treebank"] == "bulgarian") & (df_temp["Language Model"] == "mBERT")])
df_temp["Error reduction"] = df_temp.groupby(['Treebank', 'Language Model'])['Error'].diff()
df_temp = df_temp.dropna()
df_temp["Relative error reduction"] = (df_temp["Error reduction"] / df_temp["Error"]) * 100


sns.despine(right=True)
fig = plt.figure()
plt.xticks(rotation=45)
plt.ylim(-20,50)
#df_temp['order1'] = df_temp['Treebank'].map(lambda x: const_order_dict[x])
#df_temp['order2'] = df_temp['Language Model'].map(lambda x: lm_order_dic[x])
#df_temp = df_temp.sort_values(by=['order1', 'order2'])
plot = sns.barplot(x='Treebank', y='Error reduction', hue='Language Model', data=df_temp, palette="Set2").get_figure()
sns.set(font_scale=0.5, style='whitegrid')
sns.set_context('paper', font_scale=1)
#plot = sns.lineplot(x='language', y='BracketingFMeasure', hue='lm', style="pretrained", data=df, palette="Set2", markers=True).get_figure()

plt.legend(loc='best',ncol=1)
#plt.gca().xaxis.grid(False)

#plt.title('F Score for ' + encoding)


key = key.replace('/', '_')
plot.savefig('plots/' + '1const' + '.png', bbox_inches='tight', dpi=300)