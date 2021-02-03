import matplotlib.pyplot as plt
import ROOT as r
import math
import numpy as np
from ROOT import gROOT
from ROOT.Math import XYZVector
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import sys


myfile = r.TFile.Open("analysisPlots.root", "read")

#c1 = r.TCanvas("c1", "dR", 800, 800)
#c1.cd()
#histdR = myfile.Get("histdR")
#histdR.Draw()
#histdR.SetTitle("dR")
#histdR.GetXaxis().SetTitle("dR")
#histdR.GetYaxis().SetTitle("Events")
#c1.SetLogy()
#c1.SaveAs("images/dR.png")

c2 = r.TCanvas("c2", "c2", 1000, 500)
c2.Divide(2,1)
r.gPad.SetLogy()
c2.cd(1)
c2.SetLogy()
d0Fake = myfile.Get("d0Fake")
d0Other = myfile.Get("d0Other")
d0Other.Draw()
d0Fake.Draw("same")
d0Fake.SetTitle("Track d0")
d0Fake.GetXaxis().SetTitle("d0")
d0Fake.GetYaxis().SetTitle("Events")
d0Other.SetLineColor(2)
fakeEntries = d0Fake.GetEntries()
otherEntries = d0Other.GetEntries()
r.gStyle.SetOptStat(0000)
l2 = r.TLegend(0.1, 0.7, 0.4, 0.8)
l2.AddEntry(d0Fake, "dR > 0.15 (Fake)", "l")
l2.AddEntry(d0Other, "dR < 0.15 (Muon Selected)", "l")
l2.SetTextSize(0.02)
l2.Draw()
r.gPad.SetLogy()
c2.cd(2)
dZFake = myfile.Get("dZFake")
dZOther = myfile.Get("dZOther")
dZOther.Draw()
dZFake.Draw("same")
dZFake.SetTitle("Track dZ")
dZFake.GetXaxis().SetTitle("dZ")
dZFake.GetYaxis().SetTitle("Events")
dZOther.Draw("same")
dZOther.SetLineColor(2)
l2.Draw()
r.gStyle.SetOptStat(0000)
c2.SaveAs("images/d0Z.root")

c3 = r.TCanvas("c3", "c3", 1500, 500)
r.gPad.SetLogy()
c3.Divide(3,1)
c3.cd(1)
maxdedx = myfile.Get("maxdedx")
mumaxdedx = myfile.Get("mumaxdedx")
maxdedx.Draw()
maxdedx.SetTitle("Max DeDx Hit Layer")
maxdedx.GetXaxis().SetTitle("Layer")
maxdedx.GetYaxis().SetTitle("Events")
mumaxdedx.Draw("same")
mumaxdedx.SetLineColor(2)
r.gStyle.SetOptStat(0000)
l3 = r.TLegend(0.6, 0.7, 0.9, 0.8)
l3.AddEntry(d0Fake, "dR > 0.15 (Fake)", "l")
l3.AddEntry(d0Other, "dR < 0.15 (Muon Selected)", "l")
l3.SetTextSize(0.02)
l3.Draw()
c3.cd(2)
mindedx = myfile.Get("mindedx")
mumindedx = myfile.Get("mumindedx")
mumindedx.Draw()
mindedx.Draw("same")
mumindedx.SetLineColor(2)
mindedx.SetTitle("Min DeDx Hit Layer")
mindedx.GetXaxis().SetTitle("Layer")
mindedx.GetYaxis().SetTitle("Entries")
l3.Draw()
r.gStyle.SetOptStat(0000)
c3.cd(3)
fakeMinMax = myfile.Get("fakeMinMax")
muonMinMax = myfile.Get("muonMinMax")
fakeMinMax.Draw()
muonMinMax.Draw("same")
muonMinMax.SetLineColor(2)
fakeMinMax.SetTitle("Max-Min DeDx Hit")
fakeMinMax.GetXaxis().SetTitle("DeDx")
fakeMinMax.GetYaxis().SetTitle("Events")
r.gStyle.SetOptStat(0000)
l3.Draw()
c3.SaveAs("images/dEdX_MinMax.root")



























