/**CFile****************************************************************

  FileName    [testfirst.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [Getting started with ABC.]

  Synopsis    [Main functio of new command]

  Author      [xyz]
  
  Affiliation [ISI]

  Date        [Ver. 1.0. Started - June 15, 2023.]

  Revision    [$Id: bmc.c,v 1.00 2005/06/20 00:00:00 alanmi Exp $]

***********************************************************************/

// #include "base/main/main.h"
#include "base/abc/abc.h"
#include "base/main/main.h"
#include "aig/gia/giaAig.h"
#include "opt/dar/dar.h"
#include "sat/cnf/cnf.h"
#include "proof/fra/fra.h"
#include "proof/fraig/fraig.h"
#include "proof/int/int.h"
#include "proof/dch/dch.h"
#include "proof/ssw/ssw.h"
#include "opt/cgt/cgt.h"
#include "bdd/bbr/bbr.h"
#include "aig/gia/gia.h"
#include "proof/cec/cec.h"
#include "opt/csw/csw.h"
#include "proof/pdr/pdr.h"
#include "sat/bmc/bmc.h"
#include "map/mio/mio.h"

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

int TestFirst_FirstFunction(Abc_Ntk_t * pNtk);


////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Function for first command]

  Description [Extracts the ABC network and executes the main function for the command]
               
  SideEffects []

  SeeAlso     []
***********************************************************************/

int TestFirst_FirstFunctionAbc(Abc_Frame_t * pAbc){
  Abc_Ntk_t *pNtk;
  int result;

  //Gte the network that is raed into ABC
  pNtk = Abc_FrameReadNtk(pAbc);

  if (pNtk == NULL){
    Abc_Print(-1, "TestFirst_FirstFunctionAbc: Getting the target network has failed.\n");
    return 0;
  }
  // call the main function
  result = TestFirst_FirstFunction(pNtk);

  return result;
}

/**Function*************************************************************

  Synopsis    [Main function for first command]

  Description [prints information for a structurally hashed network]
               
  SideEffects []

  SeeAlso     []
***********************************************************************/

int TestFirst_FirstFunction(Abc_Ntk_t * pNtk){

  // check if the network is strashed
  if (! Abc_NtkIsStrash(pNtk)){
    Abc_Print(-1, "TestFirst_FirstFunction: This command is only applicable to strashed networks.\n");
    return 0;
  }

  // print information about the network
  Abc_Print(1, "The network with name %s has: \n", Abc_NtkName(pNtk));
  Abc_Print(1, "\t- %d primary inputs; \n", Abc_NtkPiNum(pNtk));
  Abc_Print(1, "\t- %d primary outputs; \n", Abc_NtkPoNum(pNtk));
  Abc_Print(1, "\t- %d primary gates; \n", Abc_NtkNodeNum(pNtk));

  // Aig_Man_t * pMan;
  // Vec_Int_t * vMap = NULL;
  
  // pMan = Abc_NtkToDarBmc( pNtk, &vMap );
  // Abc_Print(1, "Test %s, %d", pMan->pName, pMan->nRegs );

  return -1;
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END
