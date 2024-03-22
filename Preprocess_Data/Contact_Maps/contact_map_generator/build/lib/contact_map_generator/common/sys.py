import struct, sys, fcntl, termios, tty

def termsize():
    s=struct.pack("HHHH", 0, 0, 0, 0)
    fd_stdout=sys.stdout.fileno()
    x=fcntl.ioctl(fd_stdout, termios.TIOCGWINSZ, s)
    (rows,cols,xp,yp)=struct.unpack("HHHH", x)
    return (rows,cols)


# does not work for special keys such as F#, Home, Arrow_keys
def getc():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# works for special keys but not for ESC (need double hits)
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        buf=[]
        ch = sys.stdin.read(1)
        if ch=='\x1b':
            while not (ch in ('A','B','C','D','H','F','~','Q','q')):
                buf.append(ch)
                ch = sys.stdin.read(1)
                if ch in ('\x03','\x04',' ','\r','\x1b'): break
            buf.append(ch)
            ch=''.join(buf)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
