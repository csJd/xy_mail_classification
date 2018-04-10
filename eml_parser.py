import email
import chardet
import json
import os
import html2text


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
                if enc is None:
                    enc = 'utf-8'

                try:
                    if 'plain' in ctype:
                        txt += decoded_bytes.decode(encoding=enc)
                        flag = True
                    else:
                        html = decoded_bytes.decode(encoding=enc)
                except Exception:  # few special files parse failed
                    pass

        if not flag:
            h2t = html2text.HTML2Text()
            txt = h2t.handle(html)  # transfer html to plain text

        return txt


def main():
    """ Test the module

    :return:
    """

    with open('config.json') as f:
        config = json.load(f)

    eml_dir = config.get('eml_dir')  # url to eml files

    files = os.listdir(eml_dir)

    print('Parsing...')
    for filename in files:
        if not filename.endswith(".eml"):
            continue
        file_url = os.path.join(eml_dir, filename)
        # print("Current parsing %s." % file_url)
        parsed = parse(file_url)
        filetype = ".txt"
        filename = filename[:80]  # avoid filename too long (255bytes)
        parsed_dir = config.get('parsed_dir')
        os.makedirs(parsed_dir, exist_ok=True)  # make dirs if not exists

        parsed_url = os.path.join(parsed_dir, filename + filetype)  # url to ./data/parsed/
        with open(parsed_url, "w", encoding='utf-8') as f:
            f.write(parsed)

    print('Done.')


if __name__ == '__main__':
    main()
