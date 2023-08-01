import numpy as np

def CoarseMeyerCoeff(fhat, C, n, deg):
    # CoarseMeyerCoeff -- Resume coefficients, coarse level
    # Inputs
    #   fhat: FFT of signal vector, dyadic length
    #   C: coarse resolution level
    #   n: length of signal vector (must be of the form 2^J)
    #   deg: degree of Meyer window
    # Outputs	
    #   beta: Father Meyer wavelet coefficients, level C
    #         (length(alpha) = 2^C)

    # Set end points and separate signal into real and imag parts.
    lendp = -2 ** (C - 1)
    rendp =2 ** (C - 1)
    rfh = np.real(fhat)
    ifh = np.imag(fhat)

    # Compute DCT Coefficients, Based on Real Part of FFT of Signal
    # Do Folding Operation w/ (+,+) Polarity
    fldx = FoldMeyer(rfh, [lendp, rendp], 'pp', 'f', deg)

    # Take DCT-I of the folded signal.
    rtrigcoefs = QuasiDCT(fldx, 'f')

    # Compute DST Coefficients, Based on Imag Part of FFT of Signal
    # Do Folding Operation w/ (-,-) Polarity
    fldx = FoldMeyer(ifh, [lendp, rendp], 'mm', 'f', deg)

    # Take DST-I of the folded signal.
    itrigcoefs = QuasiDST(fldx, 'f')

    # Create coarse level wavelet coefficients for level C, from the DCT & DST Coefficients.
    beta = CombineCoeff(rtrigcoefs, itrigcoefs, 'f', n)

    return beta

def CoarseMeyerProj(beta, C, n, deg):
    # CoarseMeyerProj -- Invert Meyer Transform, coarse level C
    # Inputs
    #   beta: Father Meyer Coefficients, dyadic length 2^C.
    #   C: coarse resolution level
    #   n: length of signal vector (must be of the form 2^J)
    #   deg: degree of Meyer window
    # Outputs
    #   cpjf: projection of signal onto approximation space V_C (length(cpjf) = n)

    lendp =  -2**(C-1)
    rendp = 2**(C-1)

    # Calculate trigonometric coefficients from wavelet coefficients.
    rtrigcoefs, itrigcoefs = SeparateCoeff(beta, 'f')

    # Calculate projection of real part of \hat f (even)
    # Take DCT-I of local cosine coefficients.
    rtrigrec = QuasiDCT(rtrigcoefs, 'i')
    # Unfold trigonometric reconstruction w/ (+,+) polarity.
    unflde = UnfoldMeyer(rtrigrec, [lendp, rendp], 'pp', 'f', deg)
    # Extend unfolded signal to integers -n/2+1 -> n/2.
    eextproj = ExtendProj(unflde, n, 'f', [lendp, rendp], 'e')

    # Calculate projection of imaginary part of \hat f (odd)
    # Take DST-I of local sine coefficients.
    itrigrec = QuasiDST(itrigcoefs, 'i')
    # Unfold trigonometric reconstruction w/ (-,-) polarity.
    unfldo = UnfoldMeyer(itrigrec, [lendp, rendp], 'mm', 'f', deg)
    # Extend unfolded signal to integers -n/2+1 -> n/2.
    oextproj = ExtendProj(unfldo, n, 'f', [lendp, rendp], 'o')

    # Combine real and imaginary parts to yield coarse level projection of \hat f.
    cpjf = (eextproj + 1j * oextproj) / 2

    return cpjf

def FoldMeyer(x, sympts, polarity, window, deg):
    # FoldMeyer -- Fold a vector onto itself using a specified window
    # Inputs
    #   x: signal vector in frequency domain (typically of length n=2^J)
    #   sympts: symmetry points, of the form [a,b]
    #   polarity: string selection folding polarity
    #             'mp'  =>  (-,+)
    #             'pm'  =>  (+,-)
    #             'mm'  =>  (-,-)
    #             'pp'  =>  (+,+)
    #   window: string selecting window
    #           'm' => Mother Meyer Wavelet Window
    #           't' => Truncated Mother Meyer Wavelet Window
    #           'f' => Father Meyer Wavelet Window
    #   deg: degree of Meyer window

    pio2 = np.pi / 2

    # Window 'x' with the specified window.
    if window == 'm':
        eps = np.floor(sympts[0] / 3)
        epsp = sympts[0] - eps - 1

        lftind = np.arange(sympts[0] - eps, sympts[0]).astype(int)
        lmidind = np.arange(sympts[0] + 1, sympts[0] + eps+1).astype(int)
        rmidind = np.arange(sympts[1] - epsp, sympts[1]).astype(int)
        rghtind = np.arange(sympts[1] + 1, sympts[1] + epsp + 1).astype(int)
        lft = x[lftind] * np.sin(pio2 * WindowMeyer(3 * ((lftind - 1) / sympts[1]) - 1, deg))
        lmid = x[lmidind] * np.sin(pio2 * WindowMeyer(3 * ((lmidind - 1) / sympts[1]) - 1, deg))
        rmid = x[rmidind] * np.cos(pio2 * WindowMeyer((3 / 2) * ((rmidind) / sympts[1]) - 1, deg))
        rght = x[rghtind] * np.cos(pio2 * WindowMeyer((3 / 2) * ((rghtind) / sympts[1]) - 1, deg))

        # lftind = np.arange(sympts[0] - eps + 1, sympts[0] + 1).astype(int)
        # lmidind = np.arange(sympts[0] + 2, sympts[0] + eps + 2).astype(int)
        # rmidind = np.arange(sympts[1] - epsp + 1, sympts[1] + 1).astype(int)
        # rghtind = np.arange(sympts[1] + 2, sympts[1] + epsp + 2).astype(int)

        # lft = x[lftind] * np.sin(pio2 * WindowMeyer(3 * ((lftind - 1) / sympts[1]) - 1, deg))
        # lmid = x[lmidind] * np.sin(pio2 * WindowMeyer(3 * ((lmidind - 1) / sympts[1]) - 1, deg))
        # rmid = x[rmidind] * np.cos(pio2 * WindowMeyer((3 / 2) * ((rmidind - 1) / sympts[1]) - 1, deg))
        # rght = x[rghtind] * np.cos(pio2 * WindowMeyer((3 / 2) * ((rghtind - 1) / sympts[1]) - 1, deg))

    elif window == 'f':
        n = len(x)
        eps = np.floor(sympts[1] / 3)
        lftind = np.arange(n + sympts[0] - eps, n + sympts[0]).astype(int)
        lmidind = np.arange(n + sympts[0] + 1, n + sympts[0] + eps + 1).astype(int)
        cntrind = np.concatenate((np.arange(n + sympts[0] + eps + 2, n), np.arange(1, sympts[1] - eps + 1))).astype(int)
        rmidind = np.arange(sympts[1] - eps, sympts[1]).astype(int)
        rghtind = np.arange(sympts[1] + 1, sympts[1] + eps + 1).astype(int)

        lft = x[lftind] * np.cos(pio2 * WindowMeyer(3 * (np.abs(lftind - n - 1) / (2 * sympts[1])) - 1, deg))
        lmid = x[lmidind] * np.cos(pio2 * WindowMeyer(3 * (np.abs(lmidind - n - 1) / (2 * sympts[1])) - 1, deg))
        cntr = x[cntrind]
        rmid = x[rmidind] * np.cos(pio2 * WindowMeyer(3 * ((rmidind) / (2 * sympts[1])) - 1, deg))
        rght = x[rghtind] * np.cos(pio2 * WindowMeyer(3 * ((rghtind) / (2 * sympts[1])) - 1, deg))

    elif window == 't':
        eps = np.floor(sympts[0] / 3)
        epsp = sympts[0] - eps - 1
        lftind = np.arange(sympts[0] - eps + 1, sympts[0] + 1).astype(int)
        lmidind = np.arange(sympts[0] + 1, sympts[0] + eps + 1).astype(int)
        rmidind = np.arange(sympts[1] - epsp, sympts[1]).astype(int)
        print(lftind)
        lft = x[lftind] * np.sin(pio2 * WindowMeyer(3 * ((lftind) / sympts[1]) - 1, deg))
        lmid = x[lmidind] * np.sin(pio2 * WindowMeyer(3 * ((lmidind ) / sympts[1]) - 1, deg))
        rmid = x[rmidind]
        rght = np.zeros(len(rmidind))
    else:
        raise ValueError('Either the mother or truncated mother Meyer wavelets must be used with the polarity chosen.')

    # Fold according to the specified polarity.
    if polarity == 'mp':
        fldx = np.concatenate((-np.flipud(lft), np.flipud(rght), [0])) + np.concatenate((lmid, rmid, [x[sympts[1]]]))
    elif polarity == 'pm':
        fldx = np.concatenate(([0], np.flipud(lft), -np.flipud(rght))) + np.concatenate(([x[sympts[0]]], lmid, rmid))
    elif polarity == 'pp':
        cntr_zeros = np.zeros(len(cntrind))
        fldx = np.concatenate((x[n + sympts[0]], lmid, cntr_zeros, rmid, x[sympts[1]])) + np.concatenate(
            ([0], -np.flipud(lft), cntr, -np.flipud(rght), [0]))
    elif polarity == 'mm':
        fldx = lmid + np.concatenate((-np.flipud(lft), np.zeros(len(cntrind)), -np.flipud(rght)))
    else:
        raise ValueError('Polarity convention must be one of the following: mp, pm, mm, pp')

    return fldx

def QuasiDCT(x, dir):
    # QuasiDCT -- Nearly Discrete Cosine Transform of Type I.
    # Inputs
    #   x: signal of dyadic length
    #   dir: string direction indicator 'f' forward ; 'i' inverse
    # Outputs
    #   c: discrete cosine transform, type I, of x

    n = len(x) - 1

    # Modifications of signal and sampling of transform are different for forward and inverse Meyer wavelet transforms.
    if dir == 'f':
        x[0] = x[0] / np.sqrt(2)
        x[n] = x[n] / np.sqrt(2)
        rx = np.concatenate([x, np.zeros(n + 1)])
        y = np.concatenate([rx, np.zeros(2 * n - 2)])
        w = np.real(np.fft.fft(y))
        c = np.sqrt(2 / n) * w[0::2][:n // 2 + 1]
        c[0] = c[0] / np.sqrt(2)

    elif dir == 'i':
        x[0] = x[0] / np.sqrt(2)
        rx = np.concatenate([x, np.zeros(n + 1)])
        y = rx
        w = np.real(np.fft.fft(y))
        c = np.sqrt(2 / (n + 1)) * w[:n + 2]
        c[0] = c[0] / np.sqrt(2)
        c[n + 1] = c[n + 1] / np.sqrt(2)
    else:
        raise ValueError("Invalid direction. Use 'f' for forward and 'i' for inverse.")

    return c

def QuasiDST(x, dir):
    # QuasiDST -- Nearly Discrete Sine Transform of Type I.
    # Inputs
    #   x: signal of dyadic length
    #   dir: string direction indicator 'f' forward ; 'i' inverse
    # Outputs
    #   s: discrete sine transform, type I, of x

    n = len(x) + 1

    # Modifications of signal and sampling of transform are different for forward and inverse Meyer wavelet transforms.
    if dir == 'f':
        rx = np.concatenate([np.zeros(n - 1), x])
        y = np.concatenate([np.zeros(2), rx, np.zeros(2 * n + 1)])
        w = -1j * np.imag(np.fft.fft(y))
        s = np.sqrt(2 / n) * w[2::2][:n // 2]

    elif dir == 'i':
        rx = np.concatenate([np.zeros(n - 1), x])
        y = np.concatenate([np.zeros(1), rx, np.zeros(1)])
        w = -1j * np.imag(np.fft.fft(y))
        s = np.sqrt(2 / n) * w[1:n + 1]

    else:
        raise ValueError("Invalid direction. Use 'f' for forward and 'i' for inverse.")

    return s


def CombineCoeff(rtrigcoefs, itrigcoefs, window, n):
    """
    Combine local trigonometric coefficients into wavelet coefficients
    """
    ln = len(rtrigcoefs)
    
    if window == 'm':
        wcoefs = np.zeros(2 * ln)
        wcoefs = np.hstack([(itrigcoefs + rtrigcoefs), (itrigcoefs - rtrigcoefs)[::-1]]) / n
        wcoefs = (-1) ** np.arange(1, 2 * ln + 1) * wcoefs
    elif window == 'f':
        wcoefs = np.zeros(2 * (ln - 1))
        wcoefs[0] = rtrigcoefs[0] / n
        wcoefs[1:2 * (ln - 1) + 1] = np.hstack([(rtrigcoefs[1:ln] - itrigcoefs),
                                                (rtrigcoefs[ln - 2::-1] + itrigcoefs[ln - 3::-1])]) / (np.sqrt(2) * n)
        wcoefs = (-1) ** np.arange(0, 2 * (ln - 1)) * wcoefs
    elif window == 't':
        wcoefs = np.zeros(2 * ln)
        wcoefs = np.hstack([(itrigcoefs + rtrigcoefs), (itrigcoefs - rtrigcoefs)[::-1]]) / n
        wcoefs = (-1) ** np.arange(1, 2 * ln + 1) * wcoefs
    else:
        raise ValueError('Window given was not m, f, or t!')

    return wcoefs


def ExtendProj(proj, n, window, sympts, sym):
    # ExtendProj -- Extend a projection to all of the integers -n/2+1 -> n/2
    # Inputs
    #   proj: windowed projection vector
    #   n: length of full signal
    #   window: string selecting window used in windowing projection
    #           'm' -> mother, 'f' -> father, 't' -> truncated mother
    #   sympts: points of symmetry and antisymmetry of projection
    #   sym: string: symmetry type; 'e' -> even; 'o' -> odd
    # Outputs
    #   extproj: extended projection of length n

    if window == 'm':
        nj = sympts[1]
        frontlength = nj // 4 + nj // 12 + 1
        backlength = n // 2 - (nj + nj // 3)
        pospart = [0] * frontlength + proj.tolist() + [0] * backlength
    elif window == 'f':
        frontind = list(range((len(proj) + 1) // 2, len(proj)))
        backlength = n // 2 + 1 - len(frontind)
        pospart = proj[frontind].tolist() + [0] * backlength
    elif window == 't':
        nj1 = sympts[0]
        frontlength = n // 2 - (nj1 + nj1 // 3)
        pospart = [0] * frontlength + proj.tolist()
    else:
        raise ValueError('Window not of type m, f, or t!')

    if sym == 'e':
        extproj = pospart +  pospart[1:int(n/2-1)][::-1] #pospart[-2::-1]
    elif sym == 'o':
        extproj = pospart + [-val for val in pospart[1:int(n/2-1)][::-1]]
    else:
        raise ValueError('Even (e) or Odd (o) are the only symmetry choices!')

    return extproj


def UnfoldMeyer(x, sympts, polarity, window, deg):
    pio2 = np.pi / 2
    print("sympts:.... ", sympts)

    if window == 'm':
        eps = np.floor(sympts[0] / 3).astype(int)
        epsp = sympts[0] - eps - 1
        if polarity == 'mp':
            xi = np.arange(sympts[0] + 1, sympts[1]+1) / sympts[1]
            lft = x[:eps] * np.cos(pio2 * WindowMeyer(3 * xi[:eps] - 1, deg))
            lmid = x[:eps] * np.sin(pio2 * WindowMeyer(3 * xi[:eps] - 1, deg))
            print(eps,epsp)
            rmid = x[eps:eps + epsp] * np.cos(pio2 * WindowMeyer(1.5 * xi[eps:eps + epsp] - 1, deg))
            rght = x[eps:eps + epsp] * np.sin(pio2 * WindowMeyer(1.5 * xi[eps:eps + epsp] - 1, deg))
            unfldx = np.concatenate([-np.flip(lft), [0], lmid, rmid, [x[sympts[0]-1]], np.flip(rght)])
        elif polarity == 'pm':
            xi = np.arange(sympts[0], sympts[1]) / sympts[1]
            lft = x[1:eps + 1] * np.cos(pio2 * WindowMeyer(3 * xi[1:eps + 1] - 1, deg))
            lmid = x[1:eps + 1] * np.sin(pio2 * WindowMeyer(3 * xi[1:eps + 1] - 1, deg))
            rmid = x[eps + 1:eps + epsp + 1] * np.cos(pio2 * WindowMeyer(1.5 * xi[eps + 1:eps + epsp + 1] - 1, deg))
            rght = x[eps + 1:eps + epsp + 1] * np.sin(pio2 * WindowMeyer(1.5 * xi[eps + 1:eps + epsp + 1] - 1, deg))
            unfldx = np.concatenate([np.flip(lft), [x[0]], lmid, rmid, [0], -np.flip(rght)])
    elif window == 'f':
        n = len(x)
        eps = np.floor(sympts[1] / 3).astype(int)
        innerx = np.arange(sympts[1] - eps, sympts[1]) / (2 * sympts[1])
        outerx = np.arange(sympts[1] + 1, sympts[1] + eps + 1) / (2 * sympts[1])
        if polarity == 'pp':
            lft = np.flip(x[1:eps + 1]) * np.cos(pio2 * WindowMeyer(3 * np.flip(outerx) - 1, deg))
            lmid = x[1:eps + 1] * np.cos(pio2 * WindowMeyer(3 * np.flip(innerx) - 1, deg))
            rmid = x[2 * sympts[1] - eps:2 * sympts[1]] * np.cos(pio2 * WindowMeyer(3 * innerx - 1, deg))
            rght = np.flip(x[2 * sympts[1] - eps:2 * sympts[1]]) * np.cos(pio2 * WindowMeyer(3 * outerx - 1, deg))
            unfldx = np.concatenate([lft, [x[0]], lmid, x[eps + 1:2 * sympts[1] - eps - 1], rmid, [x[2 * sympts[1] + 1]], rght])
        elif polarity == 'mm':
            lft = np.flip(x[:eps]) * np.cos(pio2 * WindowMeyer(3 * np.flip(outerx) - 1, deg))
            lmid = x[:eps] * np.cos(pio2 * WindowMeyer(3 * np.flip(innerx) - 1, deg))
            rmid = x[2 * sympts[1] - eps:2 * sympts[1] - 1] * np.cos(pio2 * WindowMeyer(3 * innerx - 1, deg))
            rght = np.flip(x[2 * sympts[1] - eps:2 * sympts[1] - 1]) * np.cos(pio2 * WindowMeyer(3 * outerx - 1, deg))
            unfldx = np.concatenate([-lft, [0], lmid, x[eps + 1:2 * sympts[1] - eps - 1], rmid, [0], -rght])
    elif window == 't':
        eps = np.floor(sympts[0] / 3).astype(int)
        epsp = sympts[0] - eps - 1
        if polarity == 'mp':
            xi = np.arange(sympts[0] + 1, sympts[1] + 1) / sympts[1]
            lft = x[:eps] * np.cos(pio2 * WindowMeyer(3 * xi[:eps] - 1, deg))
            lmid = x[:eps] * np.sin(pio2 * WindowMeyer(3 * xi[:eps] - 1, deg))
            rmid = x[eps:eps + epsp]
            unfldx = np.concatenate([-np.flip(lft), [0], lmid, rmid, [x[eps + epsp] * np.sqrt(2)]])
        elif polarity == 'pm':
            xi = np.arange(sympts[0], sympts[1]) / sympts[1]
            lft = x[1:eps + 1] * np.cos(pio2 * WindowMeyer(3 * xi[1:eps + 1] - 1, deg))
            lmid = x[1:eps + 1] * np.sin(pio2 * WindowMeyer(3 * xi[1:eps + 1] - 1, deg))
            rmid = x[eps + 1:eps + epsp + 1]
            unfldx = np.concatenate([np.flip(lft), [x[0]], lmid, rmid, [0]])
    else:
        print('Either the mother or truncated mother Meyer wavelets must be used with the polarity chosen.')

    return unfldx



def WindowMeyer(xi, deg):
    # WindowMeyer -- auxiliary window function for Meyer wavelets.
    # Inputs
    #   xi: abscissa values for window evaluation
    #   deg: degree of the polynomial defining Nu on [0,1]
    #        1 <= deg <= 3
    # Outputs
    #   nu: polynomial of degree 'deg' if x in [0,1]
    #       1 if x > 1 and 0 if x < 0.

    if deg == 0:
        nu = xi
    elif deg == 1:
        nu = xi ** 2 * (3 - 2 * xi)
    elif deg == 2:
        nu = xi ** 3 * (10 - 15 * xi + 6 * xi ** 2)
    elif deg == 3:
        nu = xi ** 4 * (35 - 84 * xi + 70 * xi ** 2 - 20 * xi ** 3)
    else:
        raise ValueError("Degree 'deg' must be 0, 1, 2, or 3.")

    nu[xi <= 0] = 0
    nu[xi >= 1] = 1

    return nu

def shape_as_row(arr):
    return np.array(arr).flatten()

def shape_like(arr, like):
    return arr.reshape(like.shape)


def dyad(j):
    return np.arange(2 ** j)

def fft(x):
    return np.fft.fft(x, axis=0)

def ifft(x):
    return np.fft.ifft(x, axis=0)

def flipud(x):
    return np.flipud(x)

def FWT_YM(x, L, deg):
    """
    Forward Wavelet Transform (periodized Meyer Wavelet)

    Parameters:
        x: 1-d signal; length(x) = 2^J
        L: Coarsest Level of V_0; L << J
        deg: degree of polynomial window (2 <= deg <= 4)

    Returns:
        w: 1-d wavelet transform of x
    """
    y = x
    x = shape_as_row(x)
    nn = len(x)
    J = int(np.log2(nn))
    fhat = np.fft.fft(x)

    if L < 3:
        raise ValueError("L must be >= 3.")

    w = np.zeros(2**J, dtype=fhat.dtype)

    # Compute Coefficients at Coarse Level.
    w[0:2**L] = CoarseMeyerCoeff(fhat, L, nn, deg)

    # Loop to Get Detail Coefficients for levels j=L,...,J-2.
    for j in range(L, J-1):
        w[2**j:2**(j+1)] = DetailMeyerCoeff(fhat, j, nn, deg)

    # Calculate Fine Level Detail Coefficients (for j=J-1).
    w[2**(J-1):2**J] = FineMeyerCoeff(fhat, nn, deg)

    w *= (nn**0.5)
    return shape_like(w, y)


def IWT_YM(w, C, deg):
    wc = w.copy()
    if C < 3:
        C = int(input('C must be >= 3. Enter new value of C: '))
    nn = len(wc)
    J = int(np.log2(nn))
    
    # Reconstruct Projection at Coarse Level.
    beta = wc[:2**C]
    cpjf = CoarseMeyerProj(beta, C, nn, deg)
    yhat = cpjf
    
    # Loop to Get Projections at detail levels j=C,...,J-2.
    for j in range(C, J-1):
        alpha = wc[2**j:2**(j+1)]
        dpjf = DetailMeyerProj(alpha, j, 2**J, deg)
        yhat += dpjf
    
    # Calculate Projection for fine detail level, j=J-1.
    alpha = wc[2**(J-1):2**J]
    fdpjf = FineMeyerProj(alpha, J-1, 2**J, deg)
    yhat += fdpjf
    
    # Invert the transform and take the real part.
    y = (nn**0.5) * np.real(np.fft.ifft(yhat))
    
    return y



def FWT2_YM(x, deg, L):
    """
    2D Forward Wavelet Transform (periodized Meyer Wavelet)

    Parameters:
    x : numpy.ndarray
        2D signal of size (2^J, 2^J)
    deg : int
        Degree of polynomial window (2 <= deg <= 4)
    L : int
        Coarsest level for V_0; L << J

    Returns:
    w : numpy.ndarray
        2D wavelet transform of x; shape (2^J, 2^J)
    """

    if L < 3:
        raise ValueError("L must be >= 3. Enter a new value for L.")
    
    w = np.zeros_like(x)

    nn = x.shape[0]
    J = int(np.log2(nn))



    x = flipud(x)

    # Compute coefficients at Coarse Level
    cr = np.zeros((nn, 2**L))
    for m in range(nn):
        cr[m] = CoarseMeyerCoeff(fft(x[m]), L, nn, deg)

    crc = np.zeros((2**L, 2**L))
    for m in range(2**L):
        y = cr[:, m]
        w_temp = CoarseMeyerCoeff(fft(y), L, nn, deg)
        crc[:, m] = w_temp

    w[:2**L, :2**L] = flipud(crc)

    # Compute coefficients for levels j = L, ..., J-2
    for j in range(L, J-1):
        if j == L:
            dr = cr
        else:
            dr = np.zeros((nn, 2**j))
            for m in range(nn):
                dr[m] = CoarseMeyerCoeff(fft(x[m]), j, nn, deg)

        drc = np.zeros((2**j, 2**j))
        for m in range(2**j):
            y = dr[:, m]
            w_temp = DetailMeyerCoeff(fft(y), j, nn, deg)
            drc[:, m] = w_temp

        w[:2**j, 2**j:2**(j+1)] = flipud(drc)

        for m in range(nn):
            dr[m] = DetailMeyerCoeff(fft(x[m]), j, nn, deg)

        for m in range(2**j):
            y = dr[:, m]
            w_temp = CoarseMeyerCoeff(fft(y), j, nn, deg)
            drc[:, m] = w_temp

        w[2**j:2**(j+1), :2**j] = flipud(drc)

        for m in range(2**j):
            y = dr[:, m]
            w_temp = DetailMeyerCoeff(fft(y), j, nn, deg)
            drc[:, m] = w_temp

        w[2**j:2**(j+1), 2**j:2**(j+1)] = flipud(drc)

    # Compute coefficients for level j = J-1
    dr = np.zeros((nn, 2**(J-1)))
    for m in range(nn):
        dr[m] = CoarseMeyerCoeff(fft(x[m]), (J-1), nn, deg)

    drc = np.zeros((2**(J-1), 2**(J-1)))
    for m in range(2**(J-1)):
        y = dr[:, m]
        w_temp = FineMeyerCoeff(fft(y), nn, deg)
        drc[:, m] = w_temp

    w[:2**(J-1), 2**(J-1):2**J] = flipud(drc)

    for m in range(nn):
        dr[m] = FineMeyerCoeff(fft(x[m]), nn, deg)

    for m in range(2**(J-1)):
        y = dr[:, m]
        w_temp = CoarseMeyerCoeff(fft(y), (J-1), nn, deg)
        drc[:, m] = w_temp

    w[2**(J-1):2**J, :2**(J-1)] = flipud(drc)

    for m in range(2**(J-1)):
        y = dr[:, m]
        w_temp = FineMeyerCoeff(fft(y), nn, deg)
        drc[:, m] = w_temp

    w[2**(J-1):2**J, 2**(J-1):2**J] = flipud(drc)

    return nn * w

def FTWT2_YM(x, L, deg):
    n, J = x.shape
    wc = np.zeros((n, n, J, J))

    for i in range(n):
        wc[:, i, :, :] = np.transpose(FWT_YM(x[:, i], L, deg).reshape(-1, 1))

    for i in range(n):
        wc[i, :, :, :] = FWT_YM(wc[i, :, :, :], L, deg)

    return wc

def IWT2_YM(wmat, C, deg):
    if C < 3:
        C = 3

    ymat = np.zeros_like(wmat)
    nn = len(ymat[0, :])
    J = int(np.log2(nn))

    # Compute coarse level projection.
    crc = np.flipud(wmat[:2**C, :2**C])
    cr = np.zeros((nn, 2**C))
    for m in range(2**C):
        w = crc[:, m]
        y = np.real(ifft(CoarseMeyerProj(w, C, nn, deg)))
        cr[:, m] = y

    for m in range(nn):
        ymat[m, :] = np.real(ifft(CoarseMeyerProj(cr[m, :], C, nn, deg)))

    # Compute projections for levels j = C, ..., J-2.
    for j in range(C, J-1):
        # Compute horizontal contribution.
        drc = np.flipud(wmat[:2**j, 2**j:2**(j+1)])
        dr = np.zeros((nn, 2**j))
        for m in range(2**j):
            w = drc[:, m]
            y = np.real(ifft(DetailMeyerProj(w, j, nn, deg)))
            dr[:, m] = y

        for m in range(nn):
            ymat[m, :] += np.real(ifft(CoarseMeyerProj(dr[m, :], j, nn, deg)))

        # Compute vertical contribution.
        drc = np.flipud(wmat[2**j:2**(j+1), :2**j])
        for m in range(2**j):
            w = drc[:, m]
            y = np.real(ifft(CoarseMeyerProj(w, j, nn, deg)))
            dr[:, m] = y

        for m in range(nn):
            ymat[m, :] += np.real(ifft(DetailMeyerProj(dr[m, :], j, nn, deg)))

        # Compute diagonal contribution.
        drc = np.flipud(wmat[2**j:2**(j+1), 2**j:2**(j+1)])
        for m in range(2**j):
            w = drc[:, m]
            y = np.real(ifft(DetailMeyerProj(w, j, nn, deg)))
            dr[:, m] = y

        for m in range(nn):
            ymat[m, :] += np.real(ifft(DetailMeyerProj(dr[m, :], j, nn, deg)))

    # Compute projection for level j = J-1.
    # Compute horizontal contribution.
    drc = np.flipud(wmat[:2**(J-1), 2**(J-1):2**J])
    dr = np.zeros((nn, 2**(J-1)))
    for m in range(2**(J-1)):
        w = drc[:, m]
        y = np.real(ifft(FineMeyerProj(w, J-1, nn, deg)))
        dr[:, m] = y

    for m in range(nn):
        ymat[m, :] += np.real(ifft(CoarseMeyerProj(dr[m, :], J-1, nn, deg)))

    # Compute vertical contribution.
    drc = np.flipud(wmat[2**(J-1):2**J, :2**(J-1)])
    for m in range(2**(J-1)):
        w = drc[:, m]
        y = np.real(ifft(CoarseMeyerProj(w, J-1, nn, deg)))
        dr[:, m] = y

    for m in range(nn):
        ymat[m, :] += np.real(ifft(FineMeyerProj(dr[m, :], J-1, nn, deg)))

    # Compute diagonal contribution.
    drc = np.flipud(wmat[2**(J-1):2**J, 2**(J-1):2**J])
    for m in range(2**(J-1)):
        w = drc[:, m]
        y = np.real(ifft(FineMeyerProj(w, J-1, nn, deg)))
        dr[:, m] = y

    for m in range(nn):
        ymat[m, :] += np.real(ifft(FineMeyerProj(dr[m, :], J-1, nn, deg)))

    ymat = nn * np.flipud(ymat)
    return ymat

def ITWT2_YM(wc, L, deg):
    n, J = wc.shape
    x = np.zeros_like(wc)

    for i in range(n):
        x[i, :] = IWT_YM(wc[i, :], L, deg)

    for i in range(n):
        x[:, i] = IWT_YM(x[:, i], L, deg)

    return x

def DetailMeyerCoeff(fhat, j, n, deg):
    lendp = 2**(j - 1)
    rendp = 2**j

    rfh = np.real(fhat)
    ifh = np.imag(fhat)

    # Compute DST Coefficients based on Real Part of FFT of Signal
    fldx = FoldMeyer(rfh, [lendp, rendp], 'mp', 'm', deg)
    rtrigcoefs = dst_iii(fldx)

    # Compute DCT Coefficients based on Imaginary Part of FFT of Signal
    fldx = FoldMeyer(ifh, [lendp, rendp], 'pm', 'm', deg)
    itrigcoefs = dct_iii(fldx)

    # Create Wavelet Coefficients for Level j from the DST & DCT Coefficients
    alpha = CombineCoeff(rtrigcoefs, itrigcoefs, 'm', n)

    return alpha


def FineMeyerCoeff(fhat, n, deg):
    J = int(np.log2(n))
    lendp = 2**(J - 2)
    rendp = 2**(J - 1)

    rfh = np.real(fhat)
    ifh = np.imag(fhat)

    # Compute DST Coefficients based on Real Part of FFT of Signal
    print([lendp, rendp], deg)
    fldx = FoldMeyer(rfh, [lendp, rendp], 'mp', 't', deg)
    fldx[-1] = fldx[-1] / np.sqrt(2)
    rtrigcoefs = dst_iii(fldx)

    # Compute DCT Coefficients based on Imaginary Part of FFT of Signal
    fldx = FoldMeyer(ifh, [lendp, rendp], 'pm', 't', deg)
    itrigcoefs = dct_iii(fldx)

    # Create Wavelet Coefficients for Level J-1 from the DST & DCT Coefficients
    print(rtrigcoefs.shape, itrigcoefs.shape)
    falpha = CombineCoeff(rtrigcoefs, itrigcoefs, 't', n)

    return falpha


def DetailMeyerProj(alpha, j, n, deg):
    lendp = 2**(j - 1)
    rendp = 2**j

    # Calculate trigonometric coefs from wavelet coefficients
    rtrigcoefs, itrigcoefs = SeparateCoeff(alpha, 'm')

    # Calculate projection of real part of fhat (even)
    # Take DST-II of local sine coefficients
    rtrigrec = dst_ii(rtrigcoefs)

    # Unfold trigonometric reconstruction with (-,+) polarity
    unflde = UnfoldMeyer(rtrigrec, [lendp, rendp], 'mp', 'm', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    eextproj = ExtendProj(unflde, n, 'm', [lendp, rendp], 'e')

    # Calculate projection of imaginary part of fhat (odd)
    # Take DCT-II of local cosine coefficients
    itrigrec = dct_ii(itrigcoefs)

    # Unfold trigonometric reconstruction with (+,-) polarity
    unfldo = UnfoldMeyer(itrigrec, [lendp, rendp], 'pm', 'm', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    oextproj = ExtendProj(unfldo, n, 'm', [lendp, rendp], 'o')

    # Combine real and imaginary parts to yield coarse level projection of fhat
    dpjf = [ complex(eextproj[i], oextproj[i]) for i in range(len(oextproj))] #eextproj + 1j * oextproj

    return dpjf


def CoarseMeyerProj(beta, C, n, deg):
    lendp = -2**(C-1)
    rendp = 2**(C-1)

    # Calculate trigonometric coefs from wavelet coefficients
    rtrigcoefs, itrigcoefs = SeparateCoeff(beta, 'f')

    # Calculate projection of real part of fhat (even)
    # Take DCT-I of local cosine coefficients
    rtrigrec = QuasiDCT(rtrigcoefs, 'i')

    # Unfold trigonometric reconstruction with (+,+) polarity
    unflde = UnfoldMeyer(rtrigrec, [lendp, rendp], 'pp', 'f', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    eextproj = ExtendProj(unflde, n, 'f', [lendp, rendp], 'e')

    # Calculate projection of imaginary part of fhat (odd)
    # Take DST-I of local sine coefficients
    itrigrec = QuasiDST(itrigcoefs, 'i')

    # Unfold trigonometric reconstruction with (-,-) polarity
    unfldo = UnfoldMeyer(itrigrec, [lendp, rendp], 'mm', 'f', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    oextproj = ExtendProj(unfldo, n, 'f', [lendp, rendp], 'o')

    # Combine real and imaginary parts to yield coarse level projection of fhat
    cpjf = [ complex(eextproj[i], oextproj[i]) for i in range(len(oextproj))] /2 #(eextproj + 1j * oextproj) / 2

    return cpjf


def dct_ii(x):
    """
    Discrete Cosine Transform of Type II
    """
    n = len(x)
    rx = np.reshape([np.zeros(n),x], (1, 2*n), order="F")[0]
    y = np.hstack([rx, np.zeros(2 * n)])
    w = np.real(np.fft.fft(y))
    c = np.sqrt(2 / n) * w[:n]
    c[0] = c[0] / np.sqrt(2)

    return c

def dct_iii(x):
    """
    Discrete Cosine Transform of Type III
    """
    n = len(x)
    x = x.copy()
    x[0] = x[0] / np.sqrt(2)
    y = np.concatenate((x, np.zeros(3 * n)))
    w = np.fft.fft(y)
    c = np.sqrt(2 / n) * np.real(w[1:2*n:2])

    return c

def dst_i(x):
    """
    Discrete Sine Transform, Type I
    """
    n = len(x) + 1
    y = np.zeros(2 * n)
    y[1:n] = -x
    z = np.fft.fft(y)
    s = np.sqrt(2 / n) * np.imag(z[1:n])

    return s

def dst_ii(x):
    """
    Discrete Sine Transform of Type II
    """
    n = len(x)
    rx = np.reshape([np.zeros(n),x], (1, 2*n), order="F")[0]
    y = np.concatenate([rx, np.zeros(2 * n)])
    w = -1* np.imag(np.fft.fft(y))
    s = np.sqrt(2 / n) * w[1:n + 1]
    s[-1] = s[-1] / np.sqrt(2)

    return s

def dst_iii(x):
    """
    Discrete Sine Transform of Type III
    """
    n = len(x)
    x = x.copy()
    x[-1] = x[-1] / np.sqrt(2)
    y = np.concatenate((np.zeros(1), x, np.zeros(3 * n - 1)))
    w = -1*np.imag(np.fft.fft(y))
    s = np.sqrt(2 / n) * w[1:2*n:2]

    return s

def FineMeyerProj(alpha, j, n, deg):
    lendp = 2**(j-1)
    rendp = 2**j

    # Calculate trigonometric coefs from wavelet coefficients
    rtrigcoefs, itrigcoefs = SeparateCoeff(alpha, 't')

    # Calculate projection of real part of fhat (even)
    # Take DST-II of local sine coefficients
    rtrigrec = dst_ii(rtrigcoefs)

    # Unfold trigonometric reconstruction with (-,+) polarity
    unflde = UnfoldMeyer(rtrigrec, [lendp, rendp], 'mp', 't', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    eextproj = ExtendProj(unflde, n, 't', [lendp, rendp], 'e')

    # Calculate projection of imaginary part of fhat (odd)
    # Take DCT-II of local cosine coefficients
    itrigrec = dct_ii(itrigcoefs)

    # Unfold trigonometric reconstruction with (+,-) polarity
    unfldo = UnfoldMeyer(itrigrec, [lendp, rendp], 'pm', 't', deg)

    # Extend unfolded signal to integers -n/2+1 -> n/2
    oextproj = ExtendProj(unfldo, n, 't', [lendp, rendp], 'o')

    # Combine real and imaginary parts to yield coarse level projection of fhat
    # print(eextproj)

    fdpjf = [ complex(eextproj[i], oextproj[i]) for i in range(len(oextproj))] #(eextproj + 1j * oextproj)

    return fdpjf



def SeparateCoeff(wcoefs, window):
    """
    Separate wavelet coefficients into local trigonometric coefficients
    """
    nj = len(wcoefs)
    nj1 = nj // 2

    diffs = nj - nj1*2

    if window == 'm':
        rtrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (np.add(wcoefs[:nj1],wcoefs[nj1:nj-diffs][::-1])) / 2
        itrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (np.subtract(wcoefs[:nj1],wcoefs[nj1:nj-diffs][::-1])) / 2
        # rtrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (wcoefs[:nj1] + wcoefs[nj - 1:nj - nj1-1:-1]) / 2
        # itrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (wcoefs[:nj1] - wcoefs[nj - 1:nj - nj1-1:-1]) / 2
    elif window == 'f':
        rtrigcoefs = (-1) ** np.arange(1, nj) * (wcoefs[1:] + wcoefs[nj - 1:0:-1]) / np.sqrt(2)
        rtrigcoefs = np.hstack([2 * wcoefs[0], rtrigcoefs])
        itrigcoefs = (-1) ** np.arange(2, nj + 1) * (wcoefs[1:] - wcoefs[nj - 1:0:-1]) / np.sqrt(2)
    elif window == 't':
        rtrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (np.add(wcoefs[:nj1],wcoefs[nj1:nj-diffs][::-1])) / 2
        itrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (np.subtract(wcoefs[:nj1],wcoefs[nj1:nj-diffs][::-1])) / 2
        # rtrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (wcoefs[:nj1] + wcoefs[nj - 1:nj - nj1-1:-1]) / 2
        # itrigcoefs = (-1) ** np.arange(1, nj1 + 1) * (wcoefs[:nj1] - wcoefs[nj - 1:nj - nj1-1:-1]) / 2
    else:
        raise ValueError('Window given was not m, f, or t!')

    return rtrigcoefs, itrigcoefs