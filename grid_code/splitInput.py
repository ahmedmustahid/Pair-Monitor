from pyLCIO.io import LcioReader

from pyLCIO import EVENT, UTIL, IO, IOIMPL, IMPL

from math import sin, log10
import os

def run():


    direc='incoherent_pair'
    infile=direc+'/'+'incoherent_pair.slcio'

    outfile = direc+'/'+'inco_pair_split.slcio'




    reader = LcioReader.LcioReader(infile)
    totalevent = reader.getNumberOfEvents()

    wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )
    wrt.open( outfile , EVENT.LCIO.WRITE_NEW )

    newcol = 0
    newevt = 0



    for iev in range(totalevent):
        event = reader.next()

        run = IMPL.LCRunHeaderImpl()
        run.setRunNumber( iev )
        run.parameters().setValue("Generator","CAIN")
        wrt.writeRunHeader( run )

        try:
            mcps = event.getCollection('MCParticle')

            imcp=int(0)
            isubev=int(0)

            for mcp in mcps:

                if imcp%1000==0:
                    if imcp>0:
                        wrt.writeEvent( newevt )
                        isubev=isubev+1

                    newevt = IMPL.LCEventImpl()
                    newevt.setEventNumber( isubev )
                    newcol = IMPL.LCCollectionVec( EVENT.LCIO.MCPARTICLE )
                    newevt.addCollection( newcol , "MCParticle" )



                newcol.addElement( mcp )
                imcp=imcp+1

            wrt.writeEvent( newevt )
            print 'written', isubev, 'subevents'
            print outfile +' is created'
        except:
            print 'error!'



    wrt.close()


#----------------------------------

run()
