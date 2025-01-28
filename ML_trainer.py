import os, pickle, joblib
import inspect
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def write_output(results, file, class_names):
    
    output = []

    output_pdf = os.path.join(os.path.dirname(file), 'confusion_matrices.pdf')

    with PdfPages(output_pdf) as pdf:
        # Iterate over each result and create the string for output
        for name, result in results.items():
            output.append(f"\nResults for {name}:")
            
            if 'training_accuracy' in result:
                output.append(f"\nTraining Accuracy: {result['training_accuracy']}")
            else:
                output.append("\nTraining Accuracy: N/A")
            
            if 'testing_accuracy' in result:
                output.append(f"Testing Accuracy: {result['testing_accuracy']}\n")
            else:
                output.append("Testing Accuracy: N/A\n")
            
            if 'training_confusion_matrix' in result:
                output.append(f"Training Confusion Matrix:\n{result['training_confusion_matrix']}\n")
                disp = ConfusionMatrixDisplay(confusion_matrix=result['training_confusion_matrix'], display_labels=class_names)
                disp.plot(cmap='bone_r');
                plt.title(f'Training Confusion Matrix of {name}')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            else:
                output.append("Training Confusion Matrix: N/A\n")
            
            if 'testing_confusion_matrix' in result:
                output.append(f"Testing Confusion Matrix:\n{result['testing_confusion_matrix']}\n")
                disp = ConfusionMatrixDisplay(confusion_matrix=result['training_confusion_matrix'], display_labels=class_names)
                disp.plot(cmap='bone_r');
                plt.title(f'Testing Confusion Matrix of {name}')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
            else:
                output.append("Testing Confusion Matrix: N/A\n")
            
            output.append('*' * 150)
    
    with open(file, 'w') as output_file:
        output_file.writelines("\n".join(output))

def check_paths(txt_path, model_path=''):
    
    if inspect.stack()[1].function == test:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    else:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        os.makedirs(model_path, exist_ok=True)  


def train(models, X_train, y_train, verbose=2, txt=True, txt_path = os.path.join(os.getcwd(), 'results'), save_model=True, model_save_dir=os.path.join(os.getcwd(), 'models')):
    
    check_paths(txt_path, model_save_dir)

    results = {}

    for name, model in models.items():

        if verbose == 1 or verbose == 2:
            print(f"\nTraining {name}...")
        
        # For GPU models, move data to GPU memory
        if 'GPU' in name:
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
            
            # Fit the model
            model.fit(X_train_gpu, y_train_gpu)
            
            # Predictions
            y_pred_gpu = model.predict(X_train_gpu)
            
            # Convert predictions back to CPU for evaluation
            y_pred = cp.asnumpy(y_pred_gpu)

        else:
            # Standard CPU models
            model.fit(X_train, y_train)

            y_pred = model.predict(X_train)

        classes = model.classes_
        classes = classes.get() if isinstance(classes, cp.ndarray) else classes
        
        # Evaluate the model
        accuracy = accuracy_score(y_train, y_pred)
        conf_matrix = confusion_matrix(y_train, y_pred, labels=classes)
        
        results[name] = {
            'training_accuracy': accuracy,
            'training_confusion_matrix': conf_matrix
        }

        if verbose == 2:
            print(f"{name} Accuracy: {accuracy}")
            print(f"{name} Confusion Matrix:\n{conf_matrix}")
            print('*' * 150)

        if save_model:
            model_file_path = f"{model_save_dir}/{name}.pkl"

            if verbose == 2:
                print(f"Saving model {name} to {model_file_path}")
            
            # joblib.dump(model, model_file_path)
            pickle.dump(model, open(model_file_path, 'wb'))

        if verbose == 1 or verbose == 2:
                print(f"Finished...\n")

    if txt:
        write_output(results, txt_path, classes)

    else:
        return results

def train_test(models, X_train, X_test, y_train, y_test, verbose=1, txt = True, txt_path = os.path.join(os.getcwd(), 'results'), save_model = True, model_save_dir = os.path.join(os.getcwd(), 'models')):
    
    check_paths(txt_path, model_save_dir)

    results = {}
    
    for name, model in models.items():

        if verbose == 1 or verbose == 2:
            print(f"\nTraining {name}...")
        
        # For GPU models, move data to GPU memory
        if 'GPU' in name:
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
            
            # Fit the model
            model.fit(X_train_gpu, y_train_gpu)
            
            # Predictions
            y1_pred_gpu = model.predict(X_train_gpu)
            y2_pred_gpu = model.predict(X_test_gpu)
            
            # Convert predictions back to CPU for evaluation
            y1_pred = cp.asnumpy(y1_pred_gpu)
            y2_pred = cp.asnumpy(y2_pred_gpu)

        else:
            # Standard CPU models
            model.fit(X_train, y_train)

            y1_pred = model.predict(X_train)
            y2_pred = model.predict(X_test)
        
        
        classes = model.classes_
        classes = classes.get() if isinstance(classes, cp.ndarray) else classes

        
        # Evaluate the model
        accuracy1     = accuracy_score(y_train, y1_pred)
        conf_matrix1  = confusion_matrix(y_train, y1_pred, labels=classes)

        accuracy2     = accuracy_score(y_test, y2_pred)
        conf_matrix2  = confusion_matrix(y_test, y2_pred, labels=classes)
        
        results[name] = {
            'training_accuracy': accuracy1,
            'training_confusion_matrix': conf_matrix1,
            'testing_accuracy': accuracy2,
            'testing_confusion_matrix': conf_matrix2,
        }

        if verbose == 2:
            print(f"{name} Training Accuracy: {accuracy1}")
            print(f"{name} Training Confusion Matrix:\n{conf_matrix1}")
            print(f"{name} Testing Accuracy: {accuracy2}")
            print(f"{name} Testing Confusion Matrix:\n{conf_matrix2}")
            print('*' * 150)

        if save_model:

            model_file_path = f"{model_save_dir}/{name}.pkl"

            if verbose == 2:
                print(f"Saving model {name} to {model_file_path}")

            # joblib.dump(model, model_file_path)
            pickle.dump(model, open(model_file_path, 'wb'))

        if verbose == 1 or verbose == 2:
            print(f"Finished...\n")


    if txt:
        write_output(results, txt_path, classes)

    else:
        return results

def test(model_paths, X_test, y_test, verbose=2, txt=True, txt_path = os.path.join(os.getcwd(), 'results')):
    
    check_paths(txt_path)

    results = {}
    
    for m in model_paths:

        model =  pickle.load(open(m, 'rb'))

        name = os.path.basename(str(m))
        name = name.removesuffix('.pkl')

        if verbose == 1 or verbose == 2:
            print(f"\nTesting {name}...")
        
        if 'GPU' in name:
            X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
        
            # Predictions
            y_pred_gpu = model.predict(X_test_gpu)
            y_pred = cp.asnumpy(y_pred_gpu)
        
        else:
            y_pred = model.predict(X_test)

        classes = model.classes_
        classes = classes.get() if isinstance(classes, cp.ndarray) else classes
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
        
        results[name] = {
            'testing_accuracy': accuracy,
            'testing_confusion_matrix': conf_matrix
        }

        if verbose == 2:
            print(f"\n{name} Accuracy: {accuracy}")
            print(f"\n{name} Confusion Matrix:\n{conf_matrix}\n")
            print('*' * 150)

        if verbose == 1 or verbose == 2:
            print(f"Finished...\n")


    if txt:
        write_output(results, txt_path, classes)

    else:
        return results

