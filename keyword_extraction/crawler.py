import requests
import re
import csv

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
}

get_url = "https://news.cctv.com/2019/07/gaiban/cmsdatainterface/page/tech_1.jsonp?cb=tech"
res = requests.get(get_url, headers=headers)
res.encoding = 'utf-8'
text = res.text

# 使用正则表达式匹配标题、时间、链接、内容和关键词
pattern = r'"title":"(.*?)","focus_date":"(.*?)","url":"(.*?)","image":"(.*?)","brief":"(.*?)","ext_field":"(.*?)","keywords":"(.*?)"'

matches = re.findall(pattern, text)

# 将匹配结果写入CSV文件
with open('tech_news.csv', mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['标题', '时间', '内容', '来源', '编辑', '关键词', '文章链接'])
    for match in matches:
        title = match[0]
        time = match[1]
        url = match[2]
        image = match[3]
        brief = match[4]
        ext_field = match[5]
        keywords = match[6]

        response = requests.get(url)
        response.encoding = response.apparent_encoding
        html = response.text

        # 提取 author 和 source
        author_match = re.search(r'<meta\s+name="author"\s+content="([^"]*)"\s*>', html)
        source_match = re.search(r'<meta\s+name="source"\s+content="([^"]*)"\s*>', html)

        # 输出结果
        author = author_match.group(1) if author_match else None
        source = source_match.group(1) if source_match else None

        title = "《" + title + "》"
        writer.writerow([title, time, brief, source, author, keywords, url])

        print(f'标题:{title}', f'时间:{time}', f'内容:{brief}', f'来源:{source}', f'编辑:{author}',
              f'关键词:{keywords}', f'文章链接:{url}')