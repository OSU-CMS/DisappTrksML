from ROOT import TFile, TGraph, TMarker, TCanvas, TH2D, gROOT, gStyle, TLine, TLegend

gROOT.SetBatch()
gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

fin = TFile('durp.root', 'read')

h_signal = TH2D('signal', 'signal;chargino mass;log_{10}(chargino lifetime) [cm]', 9, 100, 1000, 2, 3, 5)
h_bkg = TH2D('bkg', 'signal;chargino mass;log_{10}(chargino lifetime) [cm]', 9, 100, 1000, 2, 3, 5)

h_dummy = TH2D('dummy', 'dummy', 1, 0, 1, 1, 0, 1)
h_dummy.GetXaxis().SetTitle('Background efficiency')
h_dummy.GetYaxis().SetTitle('Signal efficiency')

canvas = TCanvas("c1", "c1", 800, 800)
diagonal = TLine(0, 0, 1, 1)
diagonal.SetLineStyle(2)

for mass in range(100, 1000, 100):

    if mass != 700: continue #temporary

    for lifetime in [1000, 10000]:
        print(mass, lifetime)
        
        disc4 = fin.Get('disc_4_higgsino_%d_%d' % (mass, lifetime))
        optimal4 = fin.Get('disc_4_higgsino_%d_%d_optimal' % (mass, lifetime))
        sigma4 = fin.Get('sigma_4_higgsino_%d_%d' % (mass, lifetime))

        disc5 = fin.Get('disc_5_higgsino_%d_%d' % (mass, lifetime))
        optimal5 = fin.Get('disc_5_higgsino_%d_%d_optimal' % (mass, lifetime))
        sigma5 = fin.Get('sigma_5_higgsino_%d_%d' % (mass, lifetime))

        disc6 = fin.Get('disc_6_higgsino_%d_%d' % (mass, lifetime))
        optimal6 = fin.Get('disc_6_higgsino_%d_%d_optimal' % (mass, lifetime))
        sigma6 = fin.Get('sigma_6_higgsino_%d_%d' % (mass, lifetime))

        if mass != 700 or lifetime != 1000: continue

        disc4.SetLineColor(2)
        disc4.SetLineWidth(3)
        optimal4.SetMarkerColor(2)
        optimal4.SetMarkerSize(1.5)
        sigma4.SetLineWidth(3)
        sigma4.SetLineColor(2)
        sigma4.SetLineStyle(2)

        disc5.SetLineColor(8)
        disc5.SetLineWidth(3)
        optimal5.SetMarkerColor(8)
        optimal5.SetMarkerSize(1.5)
        sigma5.SetLineWidth(3)
        sigma5.SetLineColor(8)
        sigma5.SetLineStyle(2)

        disc6.SetLineColor(4)
        disc5.SetLineWidth(3)
        optimal6.SetMarkerColor(4)
        optimal6.SetMarkerSize(1.5)
        sigma6.SetLineWidth(3)
        sigma6.SetLineColor(4)
        sigma6.SetLineStyle(2)

        h_dummy.Draw('axis')
        diagonal.Draw('same')

        disc4.Draw('L')
        sigma4.Draw('L')
        optimal4.Draw('same')

        disc5.Draw('L')
        sigma5.Draw('L')
        optimal5.Draw('same')

        disc6.Draw('L')
        sigma6.Draw('L')
        optimal6.Draw('same')

        legend = TLegend(0.55, 0.15, 0.85, 0.45)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)

        legend.AddEntry(disc4, 'Deep Sets (nLayers = 4)', 'L')
        legend.AddEntry(sigma4, 'Fiducial map (nLayers = 4', 'L')
        legend.AddEntry(disc5, 'Deep Sets (nLayers = 5)', 'L')
        legend.AddEntry(sigma5, 'Fiducial map (nLayers = 5', 'L')
        legend.AddEntry(disc6, 'Deep Sets (nLayers #geq 6)', 'L')
        legend.AddEntry(sigma6, 'Fiducial map (nLayers #geq 6)', 'L')
        legend.Draw('same')

        canvas.SaveAs('fug.pdf')
