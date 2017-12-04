package edu.stanford.nlp.vectorlabels.learn

import Jama.Matrix
import edu.stanford.nlp.vectorlabels.core.{DenseVector, Vector}
import edu.stanford.nlp.vectorlabels.utilities.HasLogging

import scala.collection.GenSeq


/**
 *
 *
 * @author svivek
 */
trait NuclearWeightVectorUpdater[Part] extends HasLogging {

  def params: SGDParameters

  def weightStep(updateInfo: GenSeq[UpdateRecord[Part]],
                 w: Vector, A: List[Vector],
                 learningRate: Double) = {
    val wUpdates = getWeightUpdates(updateInfo, w, A)
    updateWeightVector(w, learningRate, wUpdates)

    doProxStep(w,A.size, A(0).size) // A needed to provide dimensions
  }


  def doProxStep(w: Vector, numlabels: Int, numfeats: Int) = {


    val wArr = (0 until w.size).map(i => w(i)).toArray // vector to array
    val wArr2d = (0 until numlabels).map(i => wArr.slice(0+i*numfeats, numfeats*(i+1))).toArray // 2d array numfeats*numlabels
    val wList = wArr2d.map(v => DenseVector(v: _*)) // apparently, what this does is: transform Arr[Int] to Double* and then call the apply method of DenseVector object
    val matrix = new Matrix(wArr2d) // matrix

    val svd = matrix.svd

    val u = svd.getU
    val v = svd.getV
    val sigma = svd.getS.getArray

    // clip sigma
    val clipped = clip(sigma)
    val sigmaClipped = new Matrix(clipped)

    val ANew = u.times(sigmaClipped).times(v.transpose)

    // in place update to A

    for (i <- wList.indices;
         j <- (0 until wList(i).size)) {
      wList(i)(j) = ANew.get(i, j)
    }

    // finally normalize each column of A
    (0 until wList.size).
      foreach {
        i => {
          // we want to update a_i
          val ai = wList(i)
          val norm = ai.norm

          // check that the vector still exists. If the norm is zero, we
          // are in deep trouble
          assert(!java.lang.Double.isNaN(norm) &&
            norm > 0,
            s"a_$i = $ai is NaN or zero. The norm is $norm.")

          // normalize each a_i to project it to the unit ball
          ai *= (1 / norm)
        }
      }
  }


  def clip(sigma: Array[Array[Double]]) = {
    // for now, constant step size
    val t = 0.01 * params.lambda2

    for (i <- sigma.indices) {
      val s = sigma(i)(i)

      if (s >= t) sigma(i)(i) = s - t
      else if (s <= -t) sigma(i)(i) = s + t
      else sigma(i)(i) = 0
    }
    sigma
  }


  def updateWeightVector(w: Vector, rate1: Double,
                         wUpdates: Seq[Option[Seq[(Vector, Vector)]]]) = {
    // first apply shrinkage
    w *= (1 - rate1 * params.lambda1)

    val size = wUpdates.size

    // then the gradient updates
    for (update <- wUpdates) {
      update match {
        case Some(gs: Seq[(Vector, Vector)]) =>
          gs.foreach {
            g =>
              w -= (g._1 * rate1 / size)
              w += (g._2 * rate1 / size)
          }
        case None =>
      }
    }

  }

  def getWeightUpdates(updateInfo: GenSeq[UpdateRecord[Part]],
                       w: Vector, A: List[Vector]): Seq[Option[Seq[(Vector, Vector)]]] = {
    updateInfo.map {
      up => {
        import up._
        if (loss == 0) None
        else {
          // some serious loop unrolling here to prevent un-necessary updates
          // updates only happen to features belonging to parts whose labels differ. Let's take only those
          val grads = for (partId <- 0 until x.parts.size
                           if gold.labels(partId) != prediction.labels(partId);
                           gf: Vector = gold.partLabelFeatures(partId, A);
                           pf: Vector = prediction.partLabelFeatures(partId, A)) yield
            (pf, gf)


          Some(grads)
        }
      }
    }.seq
  }
}
