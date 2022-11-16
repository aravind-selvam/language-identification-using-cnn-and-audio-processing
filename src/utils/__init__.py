import base64

def decodesound(string, filename):
    data = base64.b64decode(string)
    with open(filename, 'wb') as f:
        f.write(data)
        f.close()
