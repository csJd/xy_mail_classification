import email
import chardet
import os
import html2text
# pip install html2text


def parse(url):
    """ Parse eml file to plain string

    :param url: The url to the file to parse
    :return: Parsed string
    """

    txt = ""
    html = ""
    with open(url, 'rb') as bf:
        flag = False  # if the eml file has 'text/plain' part
        # raw_data = f.read()
        # f.seek(0)
        # fenc = chardet.detect(raw_data)

        # msg = email.message_from_file(f, encoding="utf-8")
        msg = email.message_from_binary_file(bf)

        for part in msg.walk():
            ctype = part.get_content_type()
            cdp = str(part.get('Content-Disposition'))
            if (ctype == 'text/plain' or ctype == 'text/html') and 'attachment' not in cdp:
                decoded_bytes = part.get_payload(decode=True)
                enc = part.get_content_charset()
                if enc is None:
                    enc = chardet.detect(decoded_bytes)['encoding']

                try:
                    if 'plain' in ctype:
                        txt += decoded_bytes.decode(encoding=enc)
                        flag = True
                    else:
                        html = decoded_bytes.decode(encoding=enc)
                except UnicodeError or TypeError:  # few special files parse failed
                    pass

        if not flag:
            h2t = html2text.HTML2Text()
            txt = h2t.handle(html)  # transfer html to plain text

        return txt


def main():
    """ Test the module

    :return:
    """

    dir_path = os.path.join("data", "emls")  # url to ./data/emls
    os.chdir(dir_path)
    files = os.listdir(".")

    for filename in files:
        if not filename.endswith(".eml"):
            continue
        print("Current parsing %s." % filename)
        parsed = parse(filename)
        filetype = ".txt"
        parsed_url = os.path.join("..", "parsed", filename + filetype)  # url to ./data/parsed/
        with open(parsed_url, "w", encoding='utf-8') as f:
            f.write(parsed)


if __name__ == '__main__':
    main()
