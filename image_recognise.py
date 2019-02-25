# coding=utf-8
from PIL import Image  
from PIL import ImageFilter  
import urllib
import urllib.request  
import re  
import json
import uuid
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.89 Safari/537.36"  

def baidu_stu_lookup(imageFile):  
    url = "http://stu.baidu.com/n/image?fr=html5&needRawImageUrl=true&id=WU_FILE_0&name=233.png&type=image%2Fpng&lastModifiedDate=Mon+Mar+16+2015+20%3A49%3A11+GMT%2B0800+(CST)&size="  
    raw = open(imageFile, 'rb').read()  
    url = url + str(len(raw))  
    req = urllib.request.Request(url, raw, {'Content-Type':'image/png', 'User-Agent':UA})  
    resp = urllib.request.urlopen(req)  
  
    resp_url = resp.read()      # return a pure url  
  
  
    url = "http://stu.baidu.com/n/searchpc?queryImageUrl=" + urllib.parse.quote(resp_url)  
  
    req = urllib.request.Request(url, headers={'User-Agent':UA})  
    resp = urllib.request.urlopen(req)  
  
    html = resp.read()  
    #print(html)
    fileName = "temp/%s.html" % uuid.uuid4()
    with open(fileName, "wb") as file:
        file.write(html)
    return baidu_stu_html_extract(html )  
  
  
def baidu_stu_html_extract(html):  
    #pattern = re.compile(r'<script type="text/javascript">(.*?)</script>', re.DOTALL | re.MULTILINE)  
    pattern = re.compile(b"keywords:'(.*?)'")  
    matches = pattern.findall(html)  
    if not matches:  
        return '[UNKNOWN]'  
    json_str = matches[0]  
  
    json_str = json_str.replace('\\x22', '"').replace('\\\\', '\\')  
  
    #print json_str  
  
    result = [item['keyword'] for item in json.loads(json_str)]  
  
    return '|'.join(result) if result else '[UNKNOWN]' 
if __name__ == '__main__':
    fileName = "E:\\aaaaa\\download_captcha\\temp\\curt1\\2909f2dd-5736-4706-838a-2cec2521d7d8.png"
    a=baidu_stu_lookup(fileName)
    print(a)