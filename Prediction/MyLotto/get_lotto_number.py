import requests
from bs4 import BeautifulSoup

class LottoDataset(object):
  def __init__(self, data_path):
    page_url = 'https://dhlottery.co.kr/gameResult.do?method=byWin'
    webpgage = requests.get(page_url)

    soup = BeautifulSoup(webpage.content, "html.parser")
    self.get_last_lotto_dataset(soup)

  def get_last_lotto_dataset(self, soup): 
    n_lst = [int(soup.h4.strong.string[:-1])]
    for attr in soup.find_all(attrs{'class':re.compile('ball_645 lrg ball[1-5]")}):
      n_lst.append(int(attr.string)-1) # 1 --> 0 ..... 45 --> 44
    # print("{} {}".format(n_lst[0], n_lst[1:])) # 가장 최근 회차와 해당 회차의 당첨 번호들

    lotto_df.loc[n_lst[0]-1] = n_lst
    lotto_df.to_csv(data_path, index=False)
