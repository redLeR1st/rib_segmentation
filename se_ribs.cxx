#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <functional>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"

#include "itkBinaryThresholdImageFilter.h"

#include "itkHoughTransform2DCirclesImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkThresholdImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include <list>
#include "itkCastImageFilter.h"
#include "itkMath.h"

#include "itkFlipImageFilter.h"

#include "itkTileImageFilter.h"

#include <itkSpatialObject.h>

#include <itkCropImageFilter.h>

#include "itkScaleTransform.h"

#include "itkResampleImageFilter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"
#include "itkImageDuplicator.h"

#include "itkOtsuMultipleThresholdsImageFilter.h"

#include "itkEllipseSpatialObject.h"

#include "itkRegionGrowImageFilter.h"
#include "itkNeighborhoodConnectedImageFilter.h"

#include "itkScalarImageToHistogramGenerator.h"

#include <itkOtsuMultipleThresholdsCalculator.h>

#include "vector"

#include "itkAddImageFilter.h"

#include "itkImageDuplicator.h"

#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkMedianImageFilter.h"

#include "itkConnectedComponentImageFilter.h"

#include "itkLabelShapeKeepNObjectsImageFilter.h"

#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkLabelImageToStatisticsLabelMapFilter.h"

#include "itkLabelMapToLabelImageFilter.h"

#include "itkObjectByObjectLabelMapFilter.h"

#include "itkLabelUniqueLabelMapFilter.h"

using PixelType = short;

using ImageType = itk::Image< PixelType, 3 >;
using ImageType2D = itk::Image< PixelType, 2 >;

using ReaderType = itk::ImageFileReader< ImageType >;
using WriterType = itk::ImageFileWriter< ImageType >;

typedef itk::SpatialObject<2>             SpatialObjectType;
typedef SpatialObjectType::TransformType  TransformType;
typedef   float           AccumulatorPixelType;

typedef itk::EllipseSpatialObject<2> ElipseType;

typedef itk::HoughTransform2DCirclesImageFilter<PixelType,
    AccumulatorPixelType> HoughTransformFilterType;
typedef HoughTransformFilterType::CirclesListType CirclesListType;
//typedef HoughTransformFilterType::CircleType::Type CirclesListType;

typedef itk::OtsuMultipleThresholdsImageFilter <ImageType2D, ImageType2D> OtsuFilterType2D;
typedef itk::OtsuMultipleThresholdsImageFilter <ImageType, ImageType> OtsuFilterType;

typedef itk::RegionGrowImageFilter<ImageType, ImageType> RegionGrowImageFilterType;
typedef itk::NeighborhoodConnectedImageFilter<ImageType, ImageType > ConnectedFilterType;

using ScalarImageToHistogramGeneratorType = itk::Statistics::ScalarImageToHistogramGenerator<ImageType>;
using HistogramType = ScalarImageToHistogramGeneratorType::HistogramType;

using CalculatorType = itk::OtsuMultipleThresholdsCalculator<HistogramType>;

using ConnectedComponentImageFilterType = itk::ConnectedComponentImageFilter<ImageType, ImageType>;

CalculatorType::OutputType otusHist(ScalarImageToHistogramGeneratorType::Pointer input);

OtsuFilterType::ThresholdVectorType otusImg(ReaderType::Pointer input);

ScalarImageToHistogramGeneratorType::Pointer normalize(ImageType::Pointer input);

using AddImageFilterType = itk::AddImageFilter<ImageType, ImageType>;

using DuplicatorType = itk::ImageDuplicator<ImageType>;

using LabelShapeKeepNObjectsImageFilterType = itk::LabelShapeKeepNObjectsImageFilter<ImageType>;


using ShapeLabelObjectType = itk::ShapeLabelObject<unsigned short, 3>;
using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;

using StatisticsShapeLabelObjectType = itk::StatisticsLabelObject<unsigned short, 3>;
using StatisticsLabelMapType = itk::LabelMap<StatisticsShapeLabelObjectType>;

using I2LType = itk::LabelImageToShapeLabelMapFilter<ImageType, LabelMapType>;
using I2LType_stat = itk::LabelImageToStatisticsLabelMapFilter<ImageType, StatisticsLabelMapType>;

typedef itk::LabelImageToLabelMapFilter< ImageType, LabelMapType > LabelImageToLabelMapFilterType;

using LabelMapToLabelImageFilterType = itk::LabelMapToLabelImageFilter<LabelMapType, ImageType>;

bool circle_itersect(double x1, double y1, double r1, double x2, double y2, double r2) {
    double distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    double radSumSq_m = (r1 - r2) * (r1 - r2);
    double radSumSq_p = (r1 + r2) * (r1 + r2);

    if (radSumSq_m <= distSq && distSq <= radSumSq_p) { // intersect
        return true;
    }
    else     // not intersect
        return false;
}

/**
 * Corricates the circle list globaly
 *
 * @param circles_list list of the circles 
 */
void corrigate_circles(std::vector<CirclesListType> &circles_list) {

    TransformType::OffsetType Object2ToObject1Offset;
    Object2ToObject1Offset[0] = 100;
    Object2ToObject1Offset[1] = 100;

    double av_x = 0;
    double av_y = 0;
    double av_rad = 0;

    int sum = 0;

    auto it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();
        while (itCircles != (*it).end())
        {
            av_x += (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
            av_y += (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
            av_rad += (*itCircles)->GetRadius()[0];

            sum++;
            itCircles++;
        }
        it++;
    }
    av_x /= sum;
    av_y /= sum;
    av_rad /= sum;

    std::cerr << "AVG_X " << av_x << std::endl;
    std::cerr << "AVG_Y " << av_y << std::endl;
    std::cerr << "AVG_RAD " << av_rad << std::endl;

    it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();

        // If the first circle is an outlayer we search for a better fitting circle 
        if (!circle_itersect((*itCircles)->GetObjectToParentTransform()->GetOffset()[0], (*itCircles)->GetObjectToParentTransform()->GetOffset()[1], (*itCircles)->GetRadius()[0],
            av_x, av_y, av_rad)) {

            // Mimumum searching
            int min_dist = 99999999;
            while (++itCircles != (*it).end()) {
                int x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
                int y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
                double dist = (x - av_x) * (x - av_x) + (y - av_y) * (y - av_y);

                if (min_dist > dist) {

                    CirclesListType::const_iterator first = (*it).begin();

                    Object2ToObject1Offset[0] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
                    Object2ToObject1Offset[1] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
                    (*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
                    (*first)->SetRadius((*itCircles)->GetRadius()[0]);
                    min_dist = dist;
                }
            }
        }
        it++;

    }

}

/**
 * Draws the best fitting circle on each axial slice of the image:
 * Sets the value outside the circle to zero
 *
 * @param input the input image
 * @param circles_list the list of the circles
 */
void draw_circle(ImageType::Pointer input, std::vector<CirclesListType> circles_list) {

    ImageType::IndexType localIndex3D;

    const ImageType * inputImage = input;
    ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    int tolerance = 100;
    int actual_slice_number = 0;

    auto it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();

        for (unsigned short x = 0; x < size[0]; x++) {
            for (unsigned short y = 0; y < size[1]; y++) {
                int x_cube = (x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0])*(x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0]);
                int y_cube = (y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1])*(y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1]);

                // keep only inside
                if ((x_cube + y_cube) >((*itCircles)->GetRadius()[0] + tolerance) * ((*itCircles)->GetRadius()[0]) + tolerance) {
                    localIndex3D[0] = x;
                    localIndex3D[1] = y;
                    localIndex3D[2] = actual_slice_number;
                    input->SetPixel(localIndex3D, 0);
                }
            }
        }
        actual_slice_number++;
        it++;
    }
}

/**
 * Calculates the circles on the input image.
 * Order based on accumulator image's global maxima values -> Only the first circles will be used during the drawing
 *
 * @param input 2D image to search the circles for
 * @param original 3d image
 * @param keep_only_inside if true the values outside of the circles will be set to 0
 * @param circles_list conatining the result for each axial slices of the image
 *
 * @return circles_list conatining the result for each axial slices of the image
 */
ImageType2D::Pointer do_hough_on_image(ImageType2D::Pointer input, ImageType::Pointer original, bool keep_only_inside, std::vector<CirclesListType> &circles_list) {

    typedef itk::Image< AccumulatorPixelType, 2 > AccumulatorImageType;

    ImageType2D::Pointer localImage = input;

    HoughTransformFilterType::Pointer houghFilter
        = HoughTransformFilterType::New();
    // Software Guide : EndCodeSnippet
    //  Software Guide : BeginLatex
    //
    //  We set the input of the filter to be the output of the
    //  ImageFileReader. We set also the number of circles we are looking for.
    //  Basically, the filter computes the Hough map, blurs it using a certain
    //  variance and finds maxima in the Hough map. After a maximum is found,
    //  the local neighborhood, a circle, is removed from the Hough map.
    //  SetDiscRadiusRatio() defines the radius of this disc proportional to
    //  the radius of the disc found.  The Hough map is computed by looking at
    //  the points above a certain threshold in the input image. Then, for each
    //  point, a Gaussian derivative function is computed to find the direction
    //  of the normal at that point. The standard deviation of the derivative
    //  function can be adjusted by SetSigmaGradient(). The accumulator is
    //  filled by drawing a line along the normal and the length of this line
    //  is defined by the minimum radius (SetMinimumRadius()) and the maximum
    //  radius (SetMaximumRadius()).  Moreover, a sweep angle can be defined by
    //  SetSweepAngle() (default 0.0) to increase the accuracy of detection.
    //
    //  The output of the filter is the accumulator.
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet

    houghFilter->SetInput(input);
    houghFilter->SetNumberOfCircles(10);

    houghFilter->SetMinimumRadius(10);
    houghFilter->SetMaximumRadius(30);
    houghFilter->SetSweepAngle(1);
    houghFilter->SetSigmaGradient(3);
    houghFilter->SetVariance(2);
    houghFilter->Update();


    HoughTransformFilterType::CirclesListType circles;
    circles = houghFilter->GetCircles();

    typedef  unsigned char                            OutputPixelType;
    typedef  itk::Image< PixelType, 2 > OutputImageType;
    ImageType2D::Pointer  localOutputImage = ImageType2D::New();
    ImageType2D::RegionType region;
    region.SetSize(input->GetLargestPossibleRegion().GetSize());
    region.SetIndex(input->GetLargestPossibleRegion().GetIndex());
    localOutputImage->SetRegions(region);
    localOutputImage->SetOrigin(input->GetOrigin());
    localOutputImage->SetSpacing(input->GetSpacing());
    localOutputImage->Allocate(true); 
                                     
    circles_list.push_back(circles);
    return input;
}

/**
* Corricates the circle list localy based on the average +-7 slices
*
* @param circles_list list of the circles
*/
void corrigate_circles_local(std::vector<CirclesListType> &circles_list1) {

    std::vector<CirclesListType> circles_list;

    for (int i = 0; i < circles_list1.size(); i++) {

        TransformType::OffsetType Object2ToObject1Offset;
        CirclesListType push_it;
        ElipseType::Pointer elipse = ElipseType::New();

        CirclesListType::const_iterator itCircles = circles_list1[i].begin();
        Object2ToObject1Offset[0] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
        Object2ToObject1Offset[1] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
        elipse->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
        elipse->SetRadius((*itCircles)->GetRadius());
        push_it.push_back(elipse);;
        circles_list.push_back(push_it);
    }


    TransformType::OffsetType Object2ToObject1Offset;
    Object2ToObject1Offset[0] = 100;
    Object2ToObject1Offset[1] = 100;


    int filter_size = 15;

    for (int i = 0; i < circles_list.size(); i++) {


        double av_x = 0;
        double av_y = 0;
        double av_rad = 0;

        int sum = 0;
        int j = 0;

        j = i - std::ceil(filter_size / 2);
        int end_of_filter_in_lopp = j + filter_size;
        for (; j < end_of_filter_in_lopp; j++) {
            if (j < 0 || j >= circles_list.size()) {
                continue;
            }
            CirclesListType::const_iterator itCircles = circles_list[j].begin();

            av_x += (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
            av_y += (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
            av_rad += (*itCircles)->GetRadius()[0];

            sum++;
        }
        av_x /= sum;
        av_y /= sum;
        av_rad /= sum;

        // AVG
        CirclesListType::const_iterator itCircles = circles_list[i].begin();
        double d = (((*itCircles)->GetObjectToParentTransform()->GetOffset()[0]) - (av_x))*(((*itCircles)->GetObjectToParentTransform()->GetOffset()[0]) - (av_x)) + (((*itCircles)->GetObjectToParentTransform()->GetOffset()[1]) - (av_y))*(((*itCircles)->GetObjectToParentTransform()->GetOffset()[1]) - (av_y));
        if (d < 100000) {


            CirclesListType::const_iterator first = circles_list1[i].begin();

            Object2ToObject1Offset[0] = av_x;
            Object2ToObject1Offset[1] = av_y;
            (*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
            (*first)->SetRadius(av_rad);


        }
    }
}


ImageType::Pointer close_on_3d_slices(ImageType::Pointer input, int radius, int foreground) {
    std::cout << "Closing" << std::endl;
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();

    using BinaryMorphologicalClosingImageFilterType =
        itk::BinaryMorphologicalClosingImageFilter<ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
    closingFilter->SetForegroundValue(foreground);
    closingFilter->SetKernel(structuringElement);

    closingFilter->SetInput(input);
    closingFilter->Update();

    return closingFilter->GetOutput();
}

ImageType::Pointer close_on_2d_slices(ImageType::Pointer input, int radius, int foreground) {
    std::cout << "Closing" << std::endl;
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();
    
    using BinaryMorphologicalClosingImageFilterType =
        itk::BinaryMorphologicalClosingImageFilter<ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
    closingFilter->SetForegroundValue(foreground);
    closingFilter->SetKernel(structuringElement);
    
    using PasteFilterType = itk::PasteImageFilter<ImageType, ImageType>;
    PasteFilterType::Pointer pasteFilter = PasteFilterType::New();
    
    const ImageType * inputImage = input;
    
    ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    int height_of_the_image;
    height_of_the_image = size[2];
    using ExtractFilterType = itk::ExtractImageFilter< ImageType, ImageType >;

    pasteFilter->SetDestinationImage(inputImage);
    for (int i = 1; i < height_of_the_image; i++) {
        
        ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
        extractFilter->SetDirectionCollapseToSubmatrix();
    
        size[2] = 1; // we extract along z direction
    
        ImageType::IndexType start = inputRegion.GetIndex();
    
        start[2] = i;
        ImageType::RegionType desiredRegion;
        desiredRegion.SetSize(size);
        desiredRegion.SetIndex(start);
        extractFilter->SetExtractionRegion(desiredRegion);;
        extractFilter->SetInput(inputImage);
    
        closingFilter->SetInput(extractFilter->GetOutput());
        pasteFilter->SetSourceImage(closingFilter->GetOutput());
        
        ImageType::SizeType indexRadius;
        indexRadius[0] = 1; // radius along x
        indexRadius[1] = 1; // radius along y
        indexRadius[2] = 0; // radius along z
        closingFilter->SetRadius(indexRadius);
        closingFilter->UpdateLargestPossibleRegion();
        closingFilter->Update();
        const ImageType * closingImage = closingFilter->GetOutput();
        pasteFilter->SetSourceRegion(closingImage->GetBufferedRegion());
        pasteFilter->SetDestinationIndex(start);
        
        pasteFilter->Update();
        pasteFilter->SetDestinationImage(pasteFilter->GetOutput());
    }
    std::cout << "Closing end" << std::endl;
    return pasteFilter->GetOutput();
}

// Vector of the seeds grown for left and right ribs
std::vector<ImageType::IndexType> seed_vec_global_l;
std::vector<ImageType::IndexType> seed_vec_global_r;

/**
 * Findig seed points and growing ribs or other objects from it: fins seed points on "bone image" (otsu) for regiongrowing segmentation
 *
 * @param reader_ribs image to search ribs for if @param is_cleaned_from_non_ribs is true this image should be the segmented and cleand image containing only ribs
 * @param number_of_test_file key number of the image
 * @param circles_list list of the circles
 * @param is_cleaned_from_non_ribs there if true we search seed points for labeling else we search seed points for region growing
 *
 * @return segmented image contains only the ribs
 */
ImageType::Pointer find_ribs(ImageType::Pointer reader_ribs, int number_of_test_file, std::vector<CirclesListType> circles_list, bool is_cleaned_from_non_ribs) {
    seed_vec_global_l.clear();
    seed_vec_global_r.clear();

    ImageType::Pointer image_to_process;

    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();

    typedef itk::CastImageFilter< ImageType, ImageType >
        CastingFilterType;
    CastingFilterType::Pointer caster = CastingFilterType::New();

    // Thresholding with Otsu same as vertebra
    using FilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
    FilterType::Pointer threshFilter = FilterType::New();
    threshFilter = FilterType::New();
    threshFilter->SetInput(reader_ribs);
    
    if (!is_cleaned_from_non_ribs) {
        // RIB SEARCH
        std::cout << "Thresholding for ribs" << std::endl;


        OtsuFilterType::ThresholdVectorType thresholds;
        thresholds = otusHist(normalize(reader_ribs));
        for (unsigned int i = 0; i < thresholds.size(); i++)
        {
            std::cout << thresholds[i] << std::endl;
        }
        threshFilter->SetLowerThreshold(thresholds[0]);
    }
    else {
        threshFilter->SetLowerThreshold(1);
    }
    threshFilter->SetUpperThreshold(10000);
    threshFilter->SetOutsideValue(0);
    threshFilter->SetInsideValue(1);
    threshFilter->Update();

    image_to_process = threshFilter->GetOutput();

    std::string name_of_the_file;
    name_of_the_file = std::to_string(number_of_test_file) + "_threshold_ribs.nii.gz";


    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(image_to_process);
    try
    {
        if (!is_cleaned_from_non_ribs) {
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    ImageType::IndexType index3d;
    ImageType::SizeType size_ribs = image_to_process->GetLargestPossibleRegion().GetSize();

    int offset_l_x; // Distance from the vertebra center to search the left rib 
    int offset_r_x; // Distance from the vertebra center to search the right rib

    short pixelIntensity_l = 0;
    short pixelIntensity_r = 0;

    short actual_pixel_intesity = 0;

    int end = 80;               // Length of the search area todo rename
    int width_of_the_mark = 20; // Width of the search area

    int x_c;
    int y_c;

    ImageType::IndexType seed_l;
    ImageType::IndexType seed_r;

    std::vector<ImageType::IndexType> seed_vec_l; // Stores seed points on an axial slice (left)
    std::vector<ImageType::IndexType> seed_vec_r; // Stores seed points on an axial slice (right)

    for (int z = 0; z < size_ribs[2]; z++) {
        CirclesListType::const_iterator itCircles = circles_list[z].begin();


        x_c = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];                                // Center of the circle (x axis)
        y_c = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1] - (*itCircles)->GetRadius()[0]; // Center of the circle (y axis) mius the radius
                                                                                                         // by this substraction we move the searching area a bit upper

        offset_l_x = -(*itCircles)->GetRadius()[0] - 7; // Distance from the ceneter of the circle left is 7
        offset_r_x = (*itCircles)->GetRadius()[0] + 7;  // Distance from the ceneter of the circle right is 7

        for (int x = 0; x < width_of_the_mark; x++) {
            for (int y = 0; y < size_ribs[1]; y++) {

                index3d[1] = y;
                index3d[2] = z;

                // LEFT
                index3d[0] = x_c + offset_l_x - x; // x coordinate of the pixel to be test for pixel intensity

                // if the searching is in the desired area then check for the pixel intensity
                if (x == width_of_the_mark - 1 && y_c + end > y && y > y_c) {
                    pixelIntensity_l += image_to_process->GetPixel(index3d);
                }

                if (pixelIntensity_l != 0) {

                    // set the seed point's neighbor
                    seed_l[0] = index3d[0] - 1;
                    seed_l[1] = index3d[1];
                    seed_l[2] = index3d[2];

                    // if the neighboring pixel also differs from 0 then set the seed point and add to the vector
                    if (image_to_process->GetPixel(seed_l) == 1 && x == width_of_the_mark - 1) {
                        seed_l[0]++;
                        seed_vec_l.push_back(seed_l);
                    }

                }

                // Drawing the search area
                actual_pixel_intesity = image_to_process->GetPixel(index3d);
                if (actual_pixel_intesity != 0) {
                    image_to_process->SetPixel(index3d, -1);
                }
                else {
                    if (y_c + end > y && y > y_c) {
                        image_to_process->SetPixel(index3d, -2);
                    }
                    else {
                        image_to_process->SetPixel(index3d, -3);
                    }

                }
                
                actual_pixel_intesity = 0;
                pixelIntensity_l = 0;

                // RIGHT
                index3d[0] = x_c + offset_r_x + x; // x coordinate of the pixel to be test for pixel intensity

                // if the searching is in the desired area then check for the pixel intensity
                if (x == width_of_the_mark - 1 && y_c + end > y && y > y_c) {
                    pixelIntensity_r += image_to_process->GetPixel(index3d);
                }
                
                if (pixelIntensity_r != 0) {
                
                    seed_r[0] = index3d[0] + 1;
                    seed_r[1] = index3d[1];
                    seed_r[2] = index3d[2];
                
                    // if the neighboring pixel also differs from 0 then set the seed point and add to the vector
                    if (image_to_process->GetPixel(seed_r) == 1 && x == width_of_the_mark - 1) {
                        seed_r[0]--;
                        seed_vec_r.push_back(seed_r);
                    }
                }

                // Drawing the search area
                actual_pixel_intesity = image_to_process->GetPixel(index3d);
                if (actual_pixel_intesity != 0) {
                
                    image_to_process->SetPixel(index3d, -1);
                }
                else {
                    if (y_c + end > y && y > y_c) {
                        image_to_process->SetPixel(index3d, -2);
                    }
                    else {
                        image_to_process->SetPixel(index3d, -3);
                    }
                }
                actual_pixel_intesity = 0;
                pixelIntensity_r = 0;


            }
            if (!is_cleaned_from_non_ribs) {
                // KEEP ALL SEED POINT
                // We can not decide if the seed point is belonging to a rib or not, tus we use all of them
                for (int seed_ind = 0; seed_ind < seed_vec_l.size(); seed_ind++) {
                    neighborhoodConnected->AddSeed(seed_vec_l[seed_ind]);
                    seed_vec_global_l.push_back(seed_vec_l[seed_ind]);
                }
                seed_vec_l.clear();
                
                for (int seed_ind = 0; seed_ind < seed_vec_r.size(); seed_ind++) {
                    neighborhoodConnected->AddSeed(seed_vec_r[seed_ind]);
                    seed_vec_global_r.push_back(seed_vec_r[seed_ind]);
                }
                seed_vec_r.clear();
            }
            else {
                // KEEP ONLY THE FIRST SEED POINT
                // We know that the seed point is belonging to a rib so we keep only the first one
                if (!seed_vec_l.empty()) {
                    neighborhoodConnected->AddSeed(seed_vec_l[0]);
                    seed_vec_global_l.push_back(seed_vec_l[0]);
                    seed_vec_l.clear();
                }
                if (!seed_vec_r.empty()) {
                    neighborhoodConnected->AddSeed(seed_vec_r[0]);
                    seed_vec_global_r.push_back(seed_vec_r[0]);
                    seed_vec_r.clear();
                }
            }
        }

        pixelIntensity_l = 0;
        pixelIntensity_r = 0;
    }

    
    name_of_the_file = std::to_string(number_of_test_file) + "_threshold_ribs_mark.nii.gz";

    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(image_to_process);
    try
    {
        if (!is_cleaned_from_non_ribs) {
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    neighborhoodConnected->SetInput(image_to_process);

    const PixelType lowerThreshold = 1;
    const PixelType upperThreshold = 1;

    neighborhoodConnected->SetLower(lowerThreshold);
    neighborhoodConnected->SetUpper(upperThreshold);

    ImageType::SizeType radius;
    radius[0] = 0;
    radius[1] = 0;
    radius[2] = 0;
    neighborhoodConnected->SetRadius(radius);
    
    neighborhoodConnected->SetReplaceValue(20);

    neighborhoodConnected->Update();

    caster->SetInput(neighborhoodConnected->GetOutput());
    caster->Update();
    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow.nii.gz";

    try {
        if (!is_cleaned_from_non_ribs) {
            writerT->SetFileName(name_of_the_file);
            writerT->SetInput(caster->GetOutput());
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    return caster->GetOutput();
}

/**
 * Adds img1 and img2 and returns the result
 */
ImageType::Pointer add_images(ImageType::Pointer img1, ImageType::Pointer img2) {

    ImageType::IndexType index3d;
    ImageType::SizeType size = img1->GetLargestPossibleRegion().GetSize();

    short pixelIntensity_img1 = 0;
    short pixelIntensity_img2 = 0;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                index3d[0] = x;
                index3d[1] = y;
                index3d[2] = z;
                pixelIntensity_img1 = img1->GetPixel(index3d);
                pixelIntensity_img2 = img2->GetPixel(index3d);

                if (pixelIntensity_img1 == 0) {
                    img1->SetPixel(index3d, pixelIntensity_img2);
                }
            }
        }
    }

    return img1;
}

/**
 * Draws a line on img based on the circles positions
 * 
 * @param img iamge to draw a line
 * @param index z index of the line
 * @param circles_list list of the circles
 */
void draw_line_for_vertebra(ImageType::Pointer img, ImageType::IndexType index, std::vector<CirclesListType> circles_list) {
    ImageType::IndexType index3d;
    ImageType::SizeType size = img->GetLargestPossibleRegion().GetSize();
    if (size[2] <= index[2]) {
        std::cout << "BAD VERTEBRA LINE" << std::endl;
        return;
    }
    
    index3d[2] = index[2];

    short pixelIntensity = 1;
    CirclesListType::const_iterator itCircles = circles_list[index[2]].begin();
    
    int x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0] - 40;
    
    int x_end = x + 80;

    for (; x < x_end; x++) {

        int y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1] - 40;
        int y_end = y + 80;

        for (; y < y_end; y++) {
            index3d[0] = x;
            index3d[1] = y;
            img->SetPixel(index3d, pixelIntensity);
        }
    }
}

/**
 * Labels ribs by anathomical oreder based on the seed points and draws line between vertebras
 *
 * @param ribs_only image to bel labeled
 * @param ribs_only_labeled image with the old non anathomical labels
 * @param number_of_test_file 
 * @param number_of_test_file key number of the image
 * @param circles_list list of the circles for line drawing
 */
void label_ribs(ImageType::Pointer ribs_only, ImageType::Pointer ribs_only_labeled, int number_of_test_file, std::vector<CirclesListType> circles_list) {
    std::vector<ImageType::IndexType> optimal_seed_vec_l;
    std::vector<ImageType::IndexType> optimal_seed_vec_r;

    std::vector<ImageType::IndexType> vertebra_lines_vec_l;
    std::vector<ImageType::IndexType> vertebra_lines_vec_r;

    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();

    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(ribs_only);
    duplicator->Update();

    ImageType::Pointer ribs_labeled = duplicator->GetOutput(); // Result: will contain the labeled ribs
    ribs_labeled->FillBuffer(0);

    duplicator->SetInputImage(ribs_labeled);
    duplicator->Update();

    ImageType::Pointer vertebra_lines = duplicator->GetOutput(); // Result: will contain the lines between the vertebras

    const PixelType lowerThreshold = 20;
    const PixelType upperThreshold = 20;

    neighborhoodConnected->SetLower(lowerThreshold);
    neighborhoodConnected->SetUpper(upperThreshold);

    // Neighboring pixel
    ImageType::SizeType radius;
    radius[0] = 0;
    radius[1] = 0;
    radius[2] = 0;
    neighborhoodConnected->SetRadius(radius);

    neighborhoodConnected->SetInput(ribs_only);

    int is_lapocka = false;
    int ind_l = 1;
    int ind_r = 2;

    std::set<int> added_label_values; // Used labels

    // LEFT
    for (int i = 0; i < seed_vec_global_l.size(); i++) {
        int j = 0;

        // The loop ends when the z coordinate between two seed points is not 1
        while (i + j < seed_vec_global_l.size() - 1 && ((seed_vec_global_l[i + j + 1].GetIndex()[2] - seed_vec_global_l[i + j].GetIndex()[2]) == 1) && std::abs(seed_vec_global_l[i + j + 1].GetIndex()[1] - seed_vec_global_l[i + j].GetIndex()[1]) < 19) {
            j++;
        }

        int begin = i;
        int end = i + j;

        // End index is the line betwweem the vertebra
        vertebra_lines_vec_l.push_back(seed_vec_global_l[end]);

        // We add every seed points corresponding to a label
        for (int seed_i = begin; seed_i <= end; seed_i++) {
            neighborhoodConnected->AddSeed(seed_vec_global_l[seed_i]);        
        }

        neighborhoodConnected->AddSeed(seed_vec_global_l[begin]);
        
        if (added_label_values.find(ribs_only_labeled->GetPixel(seed_vec_global_l[begin])) == added_label_values.end()) {   
            neighborhoodConnected->SetReplaceValue(ind_l);
            neighborhoodConnected->Update();

            ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
            ind_l += 2;
            added_label_values.insert(ribs_only_labeled->GetPixel(seed_vec_global_l[begin]));
        }
        neighborhoodConnected->ClearSeeds();
        i = i + j;
    }

    neighborhoodConnected->ClearSeeds();
    
    // RIGHT
    for (int i = 0; i < seed_vec_global_r.size(); i++) {
        int j = 0;

        // The loop ends when the z coordinate between two seed points is not 1
        while (i + j < seed_vec_global_r.size() - 1 && ((seed_vec_global_r[i + j + 1].GetIndex()[2] - seed_vec_global_r[i + j].GetIndex()[2]) == 1) && std::abs(seed_vec_global_r[i + j + 1].GetIndex()[1] - seed_vec_global_r[i + j].GetIndex()[1]) < 19) {
            j++;
        }

        int begin = i;
        int end = i + j;

        // End index is the line betwweem the vertebra
        vertebra_lines_vec_r.push_back(seed_vec_global_r[end]);
            
        // We add every seed points corresponding to a label
        for (int seed_i = begin; seed_i <= end; seed_i++) {
            neighborhoodConnected->AddSeed(seed_vec_global_r[seed_i]);
        }

        neighborhoodConnected->AddSeed(seed_vec_global_r[begin]);

        if (added_label_values.find(ribs_only_labeled->GetPixel(seed_vec_global_r[begin])) == added_label_values.end()) {
            neighborhoodConnected->SetReplaceValue(ind_r);
            neighborhoodConnected->Update();

            ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
            ind_r += 2;
            added_label_values.insert(ribs_only_labeled->GetPixel(seed_vec_global_r[begin]));
        }
        neighborhoodConnected->ClearSeeds();
        i = i + j;
    }

    // LINE DRAWING
    for (int i = 0; i < vertebra_lines_vec_l.size(); i++) {
         
        int avg = std::ceil((vertebra_lines_vec_l[i][2] + vertebra_lines_vec_r[i][2])/2);
        ImageType::IndexType index;
        index[2] = avg;

        draw_line_for_vertebra(vertebra_lines, index, circles_list);
    }

    // Writing output
    std::string name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_mylabel.nii.gz";
    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(ribs_labeled);
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_vert_line.nii.gz";
    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(vertebra_lines);
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }
}

/**
* Calculates and normalizes the image's histogram
* SetHistogramMin(min) drops the values under 'min'
* SetHistogramMax(max) drops the values above 'max'
* Bins: divide the range (betwween 'min' and 'max') of values into a series of intervals—and then count how many values fall into each interval.
*
* @param input image 
*
* @return normalized histogram
*/
ScalarImageToHistogramGeneratorType::Pointer normalize(ImageType::Pointer input) {

    ScalarImageToHistogramGeneratorType::Pointer scalarImageToHistogramGenerator = ScalarImageToHistogramGeneratorType::New();

    scalarImageToHistogramGenerator->SetHistogramMin(0);
    scalarImageToHistogramGenerator->SetHistogramMax(1000);
    scalarImageToHistogramGenerator->SetNumberOfBins(50);
    scalarImageToHistogramGenerator->SetAutoHistogramMinimumMaximum(false);
    scalarImageToHistogramGenerator->SetInput(input);
    scalarImageToHistogramGenerator->Compute();

    return scalarImageToHistogramGenerator;
}

/**
* Otsu threshold filter based on the image histogram
*
* @param input histogram to calculate the threshold
*
* @return thresholds provided by otsu
*/
CalculatorType::OutputType otusHist(ScalarImageToHistogramGeneratorType::Pointer input) {

    //OtsuFilterType::Pointer otsuFilter = OtsuFilterType::New();
    CalculatorType::Pointer otsuFilter = CalculatorType::New();
    //otsuFilter->SetInput(reader_original->GetOutput());
    otsuFilter->SetInputHistogram(input->GetOutput());
    otsuFilter->SetNumberOfThresholds(1);
    otsuFilter->Update(); // To compute threshold

                          //OtsuFilterType::ThresholdVectorType thresholds = otsuFilter->GetThresholds();
    const CalculatorType::OutputType & thresholds = otsuFilter->GetOutput();
    return thresholds;
}

/**
 * Otsu threshold filter based on the image
 *
 * @param input image to be thresholded
 *
 * @return thresholds provided by otsu
 */
OtsuFilterType::ThresholdVectorType otusImg(ReaderType::Pointer input) {
    
    OtsuFilterType::Pointer otsuFilter = OtsuFilterType::New();
    otsuFilter->SetInput(input->GetOutput());
    otsuFilter->SetNumberOfThresholds(1);
    otsuFilter->Update(); // To compute threshold

    OtsuFilterType::ThresholdVectorType thresholds = otsuFilter->GetThresholds();
    return thresholds;
}

/**
* Writes out the image's histogram to a .txt file
*
* @param histogram histogram to be written
* @param number_of_test_file key number of the image
*/
void write_hist(ScalarImageToHistogramGeneratorType::Pointer histogram, int number_of_test_file) {
    std::ofstream myfile;
    myfile.open(std::to_string(number_of_test_file) + "_norm_hist.txt");
    myfile << "[ ";
    for (int i = 0; i < 50; i++) {
        myfile << histogram->GetOutput()->GetFrequency(i) << " ";
    }
    myfile << "]";
    myfile.close();
}


struct Label_Princ {
    int label;
    double princ_mom;
};

struct by_princ_mom {
    bool operator()(Label_Princ const &a, Label_Princ const &b) const noexcept {
        return a.princ_mom < b.princ_mom;
    }
};

/** 
 * Labels the image containing the ribs based on its connectivity, and removes non rib objects based on their attributes
 *
 * @param ribs_only segmented images contains the result of the reginog growing
 * @param number_of_test_file key number of the image
 *
 * @return labeled image by conncentedCompontes algorithm
 */
ImageType::Pointer label_ribs_ccomponents(ImageType::Pointer ribs_only, int number_of_test_file) {

    ImageType::Pointer ret_val;

    ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New();

    connected->SetInput(ribs_only);
    connected->Update();

    std::cout << "Number of objects: " << connected->GetObjectCount() << std::endl;

    std::string name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_labed_cc.nii.gz";
    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    writerT->SetFileName(name_of_the_file);

    writerT->SetInput(connected->GetOutput());

    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    I2LType::Pointer i2l = I2LType::New();
    i2l->SetInput(connected->GetOutput());
    i2l->SetComputePerimeter(false);
    i2l->Update();

    LabelMapType * labelMap = i2l->GetOutput();

    int labelNumberBeforeFilters = labelMap->GetNumberOfLabelObjects();

    std::vector<int> remove_vec;
    std::vector<Label_Princ> Label_Princ_vec;

    // Iterate trought objects
    for (unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); n++)
    {
        std::cout << "---------------------" << std::endl;
        ShapeLabelObjectType * labelObject = labelMap->GetNthLabelObject(n);
        int label_no = itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel());
        {

            std::cout << "Label: " << label_no << std::endl;
            std::cout << "    NumberOfPixels: " << labelObject->GetNumberOfPixels() << std::endl;
            std::cout << "    Centroid: " << labelObject->GetCentroid() << std::endl;
            std::cout << "    PrincipalMoments: " << labelObject->GetPrincipalMoments() << std::endl;
            std::cout << "    Elongation: " << labelObject->GetElongation() << std::endl;

        }
        double val = labelObject->GetPrincipalMoments()[0];

        // If the object's first principal moment is higher then 50 its probably not a rib
        if (!(labelObject->GetPrincipalMoments()[0] < 50)) {
            remove_vec.push_back(label_no);
            std::cout << "The label " << label_no << " has been removed" << std::endl;
            std::cout << "++++++++++++++++++++" << std::endl;
        } else {
           std::cout << "---------------------"<<std::endl;
        }
    }

    for (int temp : remove_vec) {
        std::cout << " LABEL: " << temp << std::endl;
        labelMap->RemoveLabel(temp);
    }

    remove_vec.clear();

    LabelMapToLabelImageFilterType::Pointer labelImageConverter = LabelMapToLabelImageFilterType::New();
    labelImageConverter->SetInput(labelMap);

    std::cout << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cout << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
    std::cout << "Num of labels after filter " << labelMap->GetNumberOfLabelObjects() << std::endl;

    std::cerr << "---------------------" << std::endl;
    std::cerr << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cerr << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
    std::cerr << "Num of labels after filter " << labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cerr << "---------------------" << std::endl;

    int num_of_ribs = 24;

    // If the number of the leftover objects is higher then 24 (number of the expected ribs) we apply more filtering for the non rib objects
    // by remove the objects with too small first principal values by ordering them
    if (labelMap->GetNumberOfLabelObjects() > num_of_ribs) {
        
        std::cerr << "To much objects there!!!" << std::endl;

        for (unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); n++)
        {
            ShapeLabelObjectType * labelObject = labelMap->GetNthLabelObject(n);
            int label_no = itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel());
            //std::cout << "Label: " << label_no << std::endl;
            
            Label_Princ temp;
            temp.label = label_no;
            temp.princ_mom = labelObject->GetPrincipalMoments()[0]; 

            Label_Princ_vec.push_back(temp);
        }
        int num_of_labels = labelMap->GetNumberOfLabelObjects();
        std::sort(Label_Princ_vec.begin(), Label_Princ_vec.end(), by_princ_mom());

        for (int i = 0; i < num_of_labels - num_of_ribs; i++) {
            std::cout << " LABEL: " << Label_Princ_vec[i].label << std::endl;
            labelMap->RemoveLabel(Label_Princ_vec[i].label);
        }
        labelMap->Update();
        LabelMapToLabelImageFilterType::Pointer labelImageConverter = LabelMapToLabelImageFilterType::New();
        labelImageConverter->SetInput(labelMap);
        labelImageConverter->Update();
        writerT->SetInput(labelImageConverter->GetOutput());
        ret_val = labelImageConverter->GetOutput();

        std::cout << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cout << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
        std::cout << "Num of labels after filter FINAL " << labelMap->GetNumberOfLabelObjects() << std::endl;

        std::cerr << "---------------------" << std::endl;
        std::cerr << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cerr << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
        std::cerr << "Num of labels after filter FINAL " << labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cerr << "---------------------" << std::endl;

    }
    else {
        writerT->SetInput(labelImageConverter->GetOutput());
        ret_val = labelImageConverter->GetOutput();
    
    }

    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_labed_cc_N.nii.gz";

    writerT->SetFileName(name_of_the_file);

    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    return ret_val;
}

int main(int argc, char ** argv)
{
    std::string file;
    
    if (argv[1] != nullptr) {
        file = argv[1];
    }
    else {
        std::cerr << "The firs commandline argument should be the input file name in nii.gz format" << std::endl;
        return EXIT_SUCCESS;
    }

    int number_of_test_file = 1;

    // LOG
    std::cerr << "FILE_no: " << number_of_test_file << std::endl;
    //-------------
    std::ofstream out(std::to_string(number_of_test_file) + "_zLOG.txt");
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
    //-------------

    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();

    // Read input
    std::string original_image_file_name = file;

    ReaderType::Pointer reader_original = ReaderType::New();
    ReaderType::Pointer reader_ribs = ReaderType::New();
    reader_original->SetFileName(original_image_file_name);
    reader_ribs->SetFileName(original_image_file_name);
    try
    {
        reader_original->Update(); // Image for spine extraction without corrigation
        reader_ribs->Update(); // Image for rib labeling
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    ReaderType::Pointer reader_original_1 = ReaderType::New();
    reader_original_1->SetFileName(original_image_file_name);
    try
    {
        reader_original_1->Update(); // Image for spine extraction with 1st corrigation
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    ReaderType::Pointer reader_original_2 = ReaderType::New();
    reader_original_2->SetFileName(original_image_file_name);
    try
    {
        reader_original_2->Update(); // // Image for spine extraction with 2nd corrigation
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    // Normalization of the image histogram for Otsu Threshold filter
    ScalarImageToHistogramGeneratorType::Pointer hist_of_original_img = normalize(reader_original->GetOutput());

    // Thresholding the image based on the histogram normalized before *START*
    std::cout << "Thresholding" << std::endl;

    const CalculatorType::OutputType & thresholds = otusHist(hist_of_original_img);
    for (unsigned int i = 0; i < thresholds.size(); i++)
    {
        std::cout << thresholds[i] << std::endl;
    }

    using FilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
    FilterType::Pointer threshFilter = FilterType::New();
    threshFilter->SetInput(reader_original->GetOutput());
    threshFilter->SetLowerThreshold(thresholds[0]);
    threshFilter->SetUpperThreshold(10000);
    threshFilter->SetOutsideValue(0);
    threshFilter->SetInsideValue(1);
    threshFilter->Update();
    // Thresholding the image based on the histogram normalized before *END*
        
    // Calculating the circles on the axial slices *circles_list* will conatin the lists of circles for axial slices *START*
    const ImageType * inputImage = threshFilter->GetOutput();

    ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    int height_of_the_image;
    height_of_the_image = size[2];
    using ExtractFilterType = itk::ExtractImageFilter< ImageType, ImageType2D >;
    std::vector<CirclesListType> circles_list;
    for (int i = 0; i < height_of_the_image; i++) {

        ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
        extractFilter->SetDirectionCollapseToSubmatrix();


        size[2] = 0; // we extract along z direction

        ImageType::IndexType start = inputRegion.GetIndex();

        //const unsigned int sliceNumber = std::stoi( argv[3] );

        start[2] = i;
        ImageType::RegionType desiredRegion;
        desiredRegion.SetSize(size);
        desiredRegion.SetIndex(start);

        extractFilter->SetExtractionRegion(desiredRegion);

        extractFilter->SetInput(inputImage);

        // Calculates the circles for actual axial slice
        do_hough_on_image(extractFilter->GetOutput(), reader_original->GetOutput(), true, circles_list);

    }
    // Calculating the circles on the axial slices *circles_list* will conatin the lists of circles for axial slices *END*

    // If there is an axial slice contains zero circles, we go to the next image *STAR*
    bool zero_c_flag = false;
    auto it = circles_list.begin();
    while (it != circles_list.end())
    {
        // CirclesListType::const_iterator itCircles = (*it).begin();
        int s = (*it).size();
        // std::cout << s << std::endl;
        if (s < 1) {
            zero_c_flag = true;
            break;
        }
        it++;
    }
    if (zero_c_flag) {
        number_of_test_file++;
        return EXIT_SUCCESS;
    }
    // If there is an axial slice contains zero circles, we go to the next image *END*

        
    // Writing out the spine (hopefully) without corrigation *START*
    // Draws the circles on input iamge
    draw_circle(reader_original->GetOutput(), circles_list);
    std::string name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_0_corrig.nii.gz";

    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(reader_original->GetOutput());
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    // Writing out the spine (hopefully) without corrigation *END*

    // Writing out the spine (hopefully) with gloabl corrigations of the circles *START*
    corrigate_circles(circles_list);
    draw_circle(reader_original_1->GetOutput(), circles_list);

    name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_1_corrig.nii.gz";

    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(reader_original_1->GetOutput());
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    // Writing out the spine (hopefully) with gloabl corrigations of the circles *END*

    // Writing out the spine (hopefully) with local corrigations of the circles *START*
    corrigate_circles_local(circles_list);
    draw_circle(reader_original_2->GetOutput(), circles_list);

    name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_2_corrig.nii.gz";

    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(reader_original_2->GetOutput());
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    // Writing out the spine (hopefully) with local corrigations of the circles *END*

    // Segmentation of the ribs
    ImageType::Pointer image_ribs_only = find_ribs(reader_ribs->GetOutput(), number_of_test_file, circles_list, false); // TODO RENAME

    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(image_ribs_only);
    duplicator->Update();

    // Remove non rib shaped objects
    ImageType::Pointer image_ribs_cleaned = label_ribs_ccomponents(duplicator->GetOutput(), number_of_test_file);

    //MY LABELING
    ImageType::Pointer image_ribs_cleaned_non_labeled = find_ribs(image_ribs_cleaned, number_of_test_file, circles_list, true);

    label_ribs(image_ribs_cleaned_non_labeled, image_ribs_cleaned, number_of_test_file, circles_list);
    // Cleaning and labeling the ribs *END*


    number_of_test_file++;


    return EXIT_SUCCESS;
}