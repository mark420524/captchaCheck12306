# coding=utf-8
import os
import requests
import time
import uuid

session=requests.Session()
requests.packages.urllib3.disable_warnings()

def downloadCaptcha(dir,retry=False):
    """
    下载12306验证码图片
    :param retry:
    :return:
    """
    url = "https://kyfw.12306.cn/passport/captcha/captcha-image"
    r = session.get(url, verify=False)
    if retry:
        urlBak = "https://kyfw.12306.cn/passport/captcha/captcha-check?answer=129%2C122%2C175%2C132&login_site=E&rand=sjrand"
        time.sleep(3)
        session.get(urlBak, verify=False)
        time.sleep(3)
        r = session.get(url, verify=False)
    fileName = "%s.png" % uuid.uuid4()
    path = os.path.join(dir, fileName)
    with open(path, "wb") as file:
        file.write(r.content)
    return path
if __name__ == '__main__':
    print(downloadCaptcha("temp"))
