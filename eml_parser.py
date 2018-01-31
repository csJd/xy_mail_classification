import email
import chardet
import os


def parse(url):
    """ Parse eml file to plain string (if it has) or html

    :param url: The url to the file to parse
    :return: (flag, Parsed string) flag = True for plain text, False for html
    """
    txt = ""
    html = ""
    with open(url) as f:
        flag = False  # if the eml file has 'text/plain' part
        msg = email.message_from_file(f)
        for part in msg.walk():
            ctype = part.get_content_type()
            cdp = str(part.get('Content-Disposition'))
            if (ctype == 'text/plain' or ctype == 'text/html') and 'attachment' not in cdp:
                decoded_bytes = part.get_payload(decode=True)
                enc = chardet.detect(decoded_bytes)['encoding']
                if 'plain' in ctype:
                    txt += decoded_bytes.decode(encoding=enc)
                    flag = True
                else:
                    html = decoded_bytes.decode(encoding=enc)

        if flag:
            return flag, txt
        else:
            return flag, html


def main():
    """ Test the module

    :return:
    """
    dir_path = os.path.join("data", "emls")  # url to ./data/emls
    os.chdir(dir_path)
    files = os.listdir(".")

    for filename in files:
        print("Current parsing %s." % filename)
        flag, parsed = parse(filename)
        filetype = ".txt" if flag else ".html"
        parsed_url = os.path.join("..", "parsed", filename + filetype)  # url to ./data/parsed/
        with open(parsed_url, "w", encoding='utf-8') as f:
            f.write(parsed)


if __name__ == '__main__':
    main()
